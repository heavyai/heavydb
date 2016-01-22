package main

import (
	crand "crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/user"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	log "github.com/Sirupsen/logrus"
	"github.com/gorilla/handlers"
	"github.com/rs/cors"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"
)

var (
	port       int
	backendUrl string
	frontend   string
	dataDir    string
	readOnly   bool
	quiet      bool
	roundRobin bool
)

var (
	backendUserMap map[string]string
	backendUrls    []string
	sessionCounter int
)

type Server struct {
	Username string `json:"username"`
	Password string `json:"password"`
	Port     int    `json:"port"`
	Host     string `json:"host"`
	Database string `json:"database"`
	Master   bool   `json:"master"`
}

func getLogName(lvl string) string {
	n := filepath.Base(os.Args[0])
	h, _ := os.Hostname()
	us, _ := user.Current()
	u := us.Username
	t := time.Now().Format("20060102-150405")
	p := strconv.Itoa(os.Getpid())

	return n + "." + h + "." + u + ".log." + lvl + "." + t + "." + p
}

func init() {
	pflag.IntP("port", "p", 9092, "frontend server port")
	pflag.StringP("backend-url", "b", "", "url(s) to http-port on mapd_server, comma-delimited for multiple [http://localhost:9090]")
	pflag.StringP("frontend", "f", "frontend", "path to frontend directory")
	pflag.StringP("data", "d", "data", "path to MapD data directory")
	pflag.StringP("config", "c", "mapd.conf", "path to MapD configuration file")
	pflag.Lookup("config").NoOptDefVal = "mapd.conf"
	pflag.BoolP("read-only", "r", false, "enable read-only mode")
	pflag.BoolP("quiet", "q", false, "suppress non-error messages")
	pflag.Bool("round-robin", false, "round-robin between backend urls")
	pflag.CommandLine.MarkHidden("round-robin")

	pflag.Parse()

	viper.BindPFlag("web.port", pflag.CommandLine.Lookup("port"))
	viper.BindPFlag("web.backend-url", pflag.CommandLine.Lookup("backend-url"))
	viper.BindPFlag("web.frontend", pflag.CommandLine.Lookup("frontend"))
	viper.BindPFlag("web.round-robin", pflag.CommandLine.Lookup("round-robin"))

	viper.BindPFlag("data", pflag.CommandLine.Lookup("data"))
	viper.BindPFlag("config", pflag.CommandLine.Lookup("config"))
	viper.BindPFlag("read-only", pflag.CommandLine.Lookup("read-only"))
	viper.BindPFlag("quiet", pflag.CommandLine.Lookup("quiet"))

	viper.SetDefault("http-port", 9090)

	viper.SetEnvPrefix("MAPD")
	r := strings.NewReplacer(".", "_")
	viper.SetEnvKeyReplacer(r)
	viper.AutomaticEnv()

	viper.SetConfigType("toml")
	viper.AddConfigPath("/etc/mapd")
	viper.AddConfigPath("$HOME/.config/mapd")
	viper.AddConfigPath(".")

	if viper.IsSet("config") {
		viper.SetConfigFile(viper.GetString("config"))
		err := viper.ReadInConfig()
		if err != nil {
			log.Fatal(err)
		}
	}

	port = viper.GetInt("web.port")
	backendUrl = viper.GetString("web.backend-url")
	frontend = viper.GetString("web.frontend")
	roundRobin = viper.GetBool("web.round-robin")

	dataDir = viper.GetString("data")
	readOnly = viper.GetBool("read-only")
	quiet = viper.GetBool("quiet")

	if backendUrl == "" {
		backendUrl = "http://localhost:" + strconv.Itoa(viper.GetInt("http-port"))
	}

	backendUrls = strings.Split(backendUrl, ",")
	backendUserMap = make(map[string]string)
	sessionCounter = 0
}

func uploadHandler(rw http.ResponseWriter, r *http.Request) {
	var (
		status int
		err    error
	)

	defer func() {
		if err != nil {
			http.Error(rw, err.Error(), status)
		}
	}()

	err = r.ParseMultipartForm(32 << 20)
	if err != nil {
		status = http.StatusInternalServerError
		return
	}

	if readOnly {
		status = http.StatusUnauthorized
		err = errors.New("Uploads disabled: server running in read-only mode.")
		return
	}

	uploadDir := dataDir + "/mapd_import/"
	switch r.FormValue("uploadtype") {
	case "image":
		uploadDir = dataDir + "/mapd_images/"
	default:
		sessionId := r.Header.Get("sessionid")
		uploadDir = dataDir + "/mapd_import/" + sessionId + "/"
	}

	for _, fhs := range r.MultipartForm.File {
		for _, fh := range fhs {
			infile, err := fh.Open()
			if err != nil {
				status = http.StatusInternalServerError
				return
			}
			err = os.MkdirAll(uploadDir, 0755)
			if err != nil {
				status = http.StatusInternalServerError
				return
			}
			outfile, err := os.Create(uploadDir + fh.Filename)
			if err != nil {
				status = http.StatusInternalServerError
				return
			}
			_, err = io.Copy(outfile, infile)
			if err != nil {
				status = http.StatusInternalServerError
				return
			}
			fp := filepath.Base(outfile.Name())
			rw.Write([]byte(fp))
		}
	}
}

func deleteUploadHandler(rw http.ResponseWriter, r *http.Request) {
	// not yet implemented
}

func generateRandomBytes(n int) ([]byte, error) {
	b := make([]byte, n)
	_, err := crand.Read(b)
	if err != nil {
		return nil, err
	}
	return b, nil
}

func generateRandomString(n int) (string, error) {
	sid := ""
	sidb, err := generateRandomBytes(n)
	if err != nil {
		sid = strconv.Itoa(rand.Int())
	} else {
		sid = base64.URLEncoding.EncodeToString(sidb)
	}
	return sid, err
}

func selectBestServerRand() string {
	return backendUrls[rand.Intn(len(backendUrls))]
}

func selectBestServerRR() string {
	sessionCounter++
	return backendUrls[sessionCounter%len(backendUrls)]
}

func selectBestServer() string {
	if roundRobin {
		return selectBestServerRR()
	} else {
		return selectBestServerRand()
	}
}

func thriftOrFrontendHandler(rw http.ResponseWriter, r *http.Request) {
	h := http.StripPrefix("/", http.FileServer(http.Dir(frontend)))

	c, err := r.Cookie("session")
	if err != nil || len(c.Value) < 1 {
		sid, err := generateRandomString(32)
		if err != nil {
			log.Error("failed to generate random string: ", err)
		}
		c = &http.Cookie{Name: "session", Value: sid}
		http.SetCookie(rw, c)
	}
	s := c.Value

	be, ok := backendUserMap[s]
	if !ok {
		be = selectBestServer()
		backendUserMap[s] = be
	}

	if r.Method == "POST" {
		u, _ := url.Parse(be)
		h = httputil.NewSingleHostReverseProxy(u)
		rw.Header().Del("Access-Control-Allow-Origin")
	}

	h.ServeHTTP(rw, r)
}

func imagesHandler(rw http.ResponseWriter, r *http.Request) {
	if r.RequestURI == "/images/" {
		rw.Write([]byte(""))
		return
	}
	h := http.StripPrefix("/images/", http.FileServer(http.Dir(dataDir+"/mapd_images/")))
	h.ServeHTTP(rw, r)
}

func downloadsHandler(rw http.ResponseWriter, r *http.Request) {
	if r.RequestURI == "/downloads/" {
		rw.Write([]byte(""))
		return
	}
	h := http.StripPrefix("/downloads/", http.FileServer(http.Dir(dataDir+"/mapd_export/")))
	h.ServeHTTP(rw, r)
}

func serversHandler(rw http.ResponseWriter, r *http.Request) {
	var j []byte
	j, err := ioutil.ReadFile(frontend + "/servers.json")
	if err != nil {
		s := Server{}
		if len(backendUrls) == 1 {
			s.Master = true
		} else {
			s.Master = false
		}
		s.Username = "mapd"
		s.Password = "HyperInteractive"
		s.Database = "mapd"

		h, p, _ := net.SplitHostPort(r.Host)
		s.Port, _ = net.LookupPort("tcp", p)
		s.Host = h
		// handle IPv6 addresses
		ip := net.ParseIP(h)
		if ip != nil && ip.To4() == nil {
			s.Host = "[" + h + "]"
		}

		ss := []Server{s}
		j, _ = json.Marshal(ss)
	}
	rw.Write(j)
}

func main() {
	if _, err := os.Stat(dataDir + "/mapd_log/"); os.IsNotExist(err) {
		os.MkdirAll(dataDir+"/mapd_log/", 0755)
	}
	lf, err := os.OpenFile(dataDir+"/mapd_log/"+getLogName("ALL"), os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		log.Fatal("Error opening log file: ", err)
	}
	defer lf.Close()

	alf, err := os.OpenFile(dataDir+"/mapd_log/"+getLogName("ACCESS"), os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		log.Fatal("Error opening log file: ", err)
	}
	defer alf.Close()

	var alog io.Writer
	if quiet {
		log.SetOutput(lf)
		alog = alf
	} else {
		log.SetOutput(io.MultiWriter(os.Stdout, lf))
		alog = io.MultiWriter(os.Stdout, alf)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/upload", uploadHandler)
	mux.HandleFunc("/images/", imagesHandler)
	mux.HandleFunc("/downloads/", downloadsHandler)
	mux.HandleFunc("/deleteUpload", deleteUploadHandler)
	mux.HandleFunc("/servers.json", serversHandler)
	mux.HandleFunc("/", thriftOrFrontendHandler)

	lmux := handlers.LoggingHandler(alog, mux)
	cmux := cors.Default().Handler(lmux)
	err = http.ListenAndServe(":"+strconv.Itoa(port), cmux)
	if err != nil {
		log.Fatal("Error listening: ", err)
	}
}
