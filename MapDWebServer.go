package main

import (
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
)

var (
	port         int
	proxyBackend bool
	backendUrl   string
	frontend     string
)

func init() {
	flag.IntVar(&port, "port", 9092, "frontend server port")
	flag.BoolVar(&proxyBackend, "proxy-backend", true, "proxy mapd_http_server")
	flag.StringVar(&backendUrl, "backend-url", "http://localhost:9090", "url to mapd_http_server")
	flag.StringVar(&frontend, "frontend", "frontend", "path to frontend directory")
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

	sessionId := r.Header.Get("sessionid")
	uploadDir := "uploads/" + sessionId + "/"
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
	}

	h.ServeHTTP(rw, r)
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/upload", uploadHandler)
	mux.HandleFunc("/deleteUpload", deleteUploadHandler)
	mux.HandleFunc("/", thriftOrFrontendHandler)

	loggedMux := handlers.LoggingHandler(os.Stdout, mux)
	err := http.ListenAndServe(":"+strconv.Itoa(port), loggedMux)
	if err != nil {
		log.Fatal("Error listening: ", err)
	}
}
