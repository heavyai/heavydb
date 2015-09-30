package main

import (
	"errors"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"strconv"

	"github.com/gorilla/handlers"
	"github.com/namsral/flag"
	"github.com/rs/cors"
)

var (
	port         int
	proxyBackend bool
	backendUrl   string
	frontend     string
	images       string
	readOnly     bool
)

func init() {
	flag.IntVar(&port, "port", 9092, "frontend server port")
	flag.BoolVar(&proxyBackend, "proxy-backend", true, "proxy mapd_http_server")
	flag.StringVar(&backendUrl, "backend-url", "http://localhost:9090", "url to mapd_http_server")
	flag.StringVar(&frontend, "frontend", "frontend", "path to frontend directory")
	flag.StringVar(&images, "images", "images", "path to images directory")
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

	uploadDir := "uploads/"
	switch r.FormValue("uploadtype") {
	case "image":
		uploadDir = images + "/"
	default:
		sessionId := r.Header.Get("sessionid")
		uploadDir = "uploads/" + sessionId + "/"
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
			fp, _ := filepath.Abs(outfile.Name())
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
	h := http.StripPrefix("/images/", http.FileServer(http.Dir(images)))
	h.ServeHTTP(rw, r)
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/upload", uploadHandler)
	mux.HandleFunc("/images/", imagesHandler)
	mux.HandleFunc("/deleteUpload", deleteUploadHandler)
	mux.HandleFunc("/", thriftOrFrontendHandler)

	lmux := handlers.LoggingHandler(os.Stdout, mux)
	cmux := cors.Default().Handler(lmux)
	err := http.ListenAndServe(":"+strconv.Itoa(port), cmux)
	if err != nil {
		log.Fatal("Error listening: ", err)
	}
}
