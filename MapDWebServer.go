package main

import (
	"encoding/json"
	"errors"
	"io"
	"io/ioutil"
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
	pflag.StringP("backend-url", "b", "", "url to http-port on mapd_server [http://localhost:9090]")
	pflag.StringP("frontend", "f", "frontend", "path to frontend directory")
	pflag.StringP("data", "d", "data", "path to MapD data directory")
	pflag.StringP("config", "c", "mapd.conf", "path to MapD configuration file")
	pflag.BoolP("read-only", "r", false, "enable read-only mode")
	pflag.BoolP("quiet", "q", false, "suppress non-error messages")

	pflag.Parse()

	viper.BindPFlag("web.port", pflag.CommandLine.Lookup("port"))
	viper.BindPFlag("web.backend-url", pflag.CommandLine.Lookup("backend-url"))
	viper.BindPFlag("web.frontend", pflag.CommandLine.Lookup("frontend"))

	viper.BindPFlag("data", pflag.CommandLine.Lookup("data"))
	viper.BindPFlag("config", pflag.CommandLine.Lookup("config"))
	viper.BindPFlag("read-only", pflag.CommandLine.Lookup("read-only"))
	viper.BindPFlag("quiet", pflag.CommandLine.Lookup("quiet"))

	viper.SetDefault("http-port", 9090)

	viper.SetEnvPrefix("MAPD")
	r := strings.NewReplacer(".", "_")
	viper.SetEnvKeyReplacer(r)
	viper.AutomaticEnv()

	viper.SetConfigFile(viper.GetString("config"))
	viper.SetConfigType("toml")
	viper.AddConfigPath("/etc/mapd")
	viper.AddConfigPath("$HOME/.config/mapd")
	viper.AddConfigPath(".")

	_ = viper.ReadInConfig()

	port = viper.GetInt("web.port")
	backendUrl = viper.GetString("web.backend-url")
	frontend = viper.GetString("web.frontend")
	dataDir = viper.GetString("data")
	readOnly = viper.GetBool("read-only")
	quiet = viper.GetBool("quiet")

	if backendUrl == "" {
		backendUrl = "http://localhost:" + strconv.Itoa(viper.GetInt("http-port"))
	}
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

func thriftOrFrontendHandler(rw http.ResponseWriter, r *http.Request) {
	h := http.StripPrefix("/", http.FileServer(http.Dir(frontend)))

	if r.Method == "POST" {
		u, _ := url.Parse(backendUrl)
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
		s.Master = true
		s.Username = "mapd"
		s.Password = "HyperInteractive"
		s.Database = "mapd"

		hp := strings.Split(r.Host, ":")
		s.Host = hp[0]
		if len(hp) > 1 {
			s.Port, _ = strconv.Atoi(hp[1])
		} else if r.URL.Scheme == "https" {
			s.Port = 443
		} else {
			s.Port = 80
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
