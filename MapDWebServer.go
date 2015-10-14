package main

import (
	"errors"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/user"
	"path/filepath"
	"strconv"
	"time"

	log "github.com/Sirupsen/logrus"
	"github.com/gorilla/handlers"
	"github.com/namsral/flag"
	"github.com/rs/cors"
)

var (
	port         int
	proxyBackend bool
	backendUrl   string
	frontend     string
	dataDir      string
	readOnly     bool
)

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
	flag.IntVar(&port, "port", 9092, "frontend server port")
	flag.BoolVar(&proxyBackend, "proxy-backend", true, "proxy mapd_http_server")
	flag.StringVar(&backendUrl, "backend-url", "http://localhost:9090", "url to mapd_http_server")
	flag.StringVar(&frontend, "frontend", "frontend", "path to frontend directory")
	flag.StringVar(&dataDir, "data", "data", "path to MapD data directory")
	flag.BoolVar(&readOnly, "read-only", false, "enable read-only mode")
	flag.Parse()
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

	if proxyBackend && r.Method == "POST" {
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

func main() {
	lf, err := os.OpenFile(dataDir+"/mapd_log/"+getLogName("ALL"), os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		log.Fatal("Error opening log file: ", err)
	}
	defer lf.Close()
	log.SetOutput(io.MultiWriter(os.Stdout, lf))

	alf, err := os.OpenFile(dataDir+"/mapd_log/"+getLogName("ACCESS"), os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		log.Fatal("Error opening log file: ", err)
	}
	defer alf.Close()

	mux := http.NewServeMux()
	mux.HandleFunc("/upload", uploadHandler)
	mux.HandleFunc("/images/", imagesHandler)
	mux.HandleFunc("/downloads/", downloadsHandler)
	mux.HandleFunc("/deleteUpload", deleteUploadHandler)
	mux.HandleFunc("/", thriftOrFrontendHandler)

	lmux := handlers.LoggingHandler(io.MultiWriter(os.Stdout, alf), mux)
	cmux := cors.Default().Handler(lmux)
	err = http.ListenAndServe(":"+strconv.Itoa(port), cmux)
	if err != nil {
		log.Fatal("Error listening: ", err)
	}
}
