package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httputil"
	"net/http/pprof"
	"net/url"
	"os"
	"os/user"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"text/template"
	"time"

	"github.com/Jeffail/gabs"
	"github.com/andrewseidl/viper"
	"github.com/gorilla/handlers"
	"github.com/gorilla/sessions"
	metrics "github.com/rcrowley/go-metrics"
	"github.com/rs/cors"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/pflag"
	"github.com/xeipuuv/gojsonschema"
	"go.etcd.io/etcd/client"
	"go.etcd.io/etcd/pkg/types"
	graceful "gopkg.in/tylerb/graceful.v1"
)

var (
	port                       int
	httpsRedirectPort          int
	backendURL                 *url.URL
	jupyterURL                 *url.URL
	jupyterPrefix              string
	frontend                   string
	serversJSON                string
	dataDir                    string
	tmpDir                     string
	certFile                   string
	peerCertFile               string
	keyFile                    string
	docsDir                    string
	readOnly                   bool
	verbose                    bool
	enableHTTPS                bool
	enableHTTPSAuth            bool
	enableHTTPSRedirect        bool
	profile                    bool
	compress                   bool
	enableMetrics              bool
	stripOmniSciUsernameHeader bool
	connTimeout                time.Duration
	version                    string
	proxies                    []reverseProxy
)

var (
	registry          metrics.Registry
	sessionStore      *sessions.CookieStore
	serversJSONParams []string
)

type server struct {
	Username string `json:"username"`
	Password string `json:"password"`
	Port     int    `json:"port"`
	Host     string `json:"host"`
	Database string `json:"database"`
	Master   bool   `json:"master"`
}

type thriftMethodTimings struct {
	Regex  *regexp.Regexp
	Start  string
	Units  string
	Labels []string
}

type reverseProxy struct {
	Path   string
	Target *url.URL
}

var (
	thriftMethodMap map[string]thriftMethodTimings
)

const (
	// The name of the cookie that holds the client Thrift session ID
	clientSessionCookieName = "omnisci_client_session_id"
	// The name of the cookie that holds the real session ID from SAML login
	thriftSessionCookieName = "omnisci_session"
	// The name of the JavaScript visible cookie indicating SAML auth has succeeded
	samlAuthCookieName = "omnisci_saml_authorized"
	// The magic value used as the "fake" session ID when Immerse is operating in SAML mode
	samlPlaceholderSessionID = "8f61e7d0-b515-49d9-ad77-37ed6e2868ea"
	// The page to redirect the user to when there are errors with SAML auth
	samlErrorPage = "/saml-error.html"
)

type stringDictionary struct {
	NodeID   string `json:"nodeId"`
	Path     string `json:"data"`
	Port     int    `json:"port"`
	HostName string `json:"hostname"`
}

type leaf struct {
	NodeID           string `json:"nodeId"`
	Path             string `json:"data"`
	Port             int    `json:"port"`
	StringDictionary string `json:"stringDictionary"`
	HostName         string `json:"hostname"`
}

type aggregator struct {
	NodeID           string   `json:"nodeId"`
	Path             string   `json:"data"`
	Port             int      `json:"port"`
	StringDictionary string   `json:"stringDictionary"`
	Leaves           []string `json:"leaves"`
	HostName         string   `json:"hostname"`
	LeafConnTimeout  int      `json:"leaf_conn_timeout"`
	LeafRecvTimeout  int      `json:"leaf_recv_timeout"`
	LeafSendTimeout  int      `json:"leaf_send_timeout"`
}

type cluster struct {
	HostNames        []string          `json:"hostnames"`
	StringDictionary *stringDictionary `json:"stringDictionary"`
	Leaves           []leaf            `json:"leaves"`
	Aggregator       *aggregator       `json:"aggregator"`
}

// Keep this in synch with the duplicate in
// LeafHostInfo.cpp
const (
	clusterSchema = "{" +
		"    \"$id\": \"https://example.com/arrays.schema.json\"," +
		"    \"$schema\": \"http://json-schema.org/draft-07/schema#\"," +
		"    \"description\": \"The json schema for the cluster file\"," +
		"    \"type\": \"object\"," +
		"    \"additionalProperties\": false," +
		"    \"properties\": {" +
		"        \"hostnames\": {" +
		"            \"type\": \"array\"," +
		"            \"items\": {" +
		"                \"type\": \"string\"" +
		"            }" +
		"        }," +
		"        \"dynamic\": {" +
		"            \"type\": \"boolean\"" +
		"        }," +
		"        \"aggregator\": {" +
		"            \"type\": \"object\"," +
		"            \"$ref\": \"#/definitions/aggregator\"" +
		"        }," +
		"        \"stringDictionary\": {" +
		"            \"type\": \"object\"," +
		"            \"$ref\": \"#/definitions/stringDictionary\"" +
		"        }," +
		"        \"leaves\": {" +
		"            \"type\": \"array\"," +
		"            \"items\": {" +
		"                \"$ref\": \"#/definitions/leaf\"" +
		"            }," +
		"            \"description\": \"The leaves on this host\"" +
		"        }" +
		"    }," +
		"    \"definitions\": {" +
		"        \"stringDictionary\": {" +
		"            \"type\": \"object\"," +
		"            \"required\": [" +
		"                \"data\"," +
		"                \"port\"" +
		"            ]," +
		"            \"additionalProperties\": false," +
		"            \"properties\": {" +
		"                \"nodeId\": {" +
		"                    \"type\": \"string\"," +
		"                    \"description\": \"The nodeId on the cluster\"" +
		"                }," +
		"                \"data\": {" +
		"                    \"type\": \"string\"," +
		"                    \"description\": \"Base path for storage\"" +
		"                }," +
		"                \"port\": {" +
		"                    \"type\": \"integer\"," +
		"                    \"description\": \"Port number to bind to\"" +
		"                }," +
		"                \"hostname\": {" +
		"                    \"type\": \"string\"," +
		"                    \"description\": \"The host name where this node will run\"" +
		"                }," +

		"                \"ssl_cert_file\": {" +
		"                    \"type\": \"string\"," +
		"                    \"description\": \"The ssl certificate file location\"" +
		"                }," +
		"                \"ssl_key_file\": {" +
		"                    \"type\": \"string\"," +
		"                    \"description\": \"The ssl key file location\"" +
		"                }," +
		"                \"cache-string-hash\": {" +
		"                    \"type\": \"boolean\"," +
		"                    \"description\": \"Enable cache to store hashes in string dictionary server\"" +
		"                }" +
		"            }" +
		"        }," +
		"        \"leaf\": {" +
		"            \"type\": \"object\"," +
		"            \"required\": [" +
		"                \"data\"," +
		"                \"port\"" +
		"            ]," +
		"            \"additionalProperties\": false," +
		"            \"properties\": {" +
		"                \"nodeId\": {" +
		"                    \"type\": \"string\"," +
		"                    \"description\": \"The nodeId on the cluster\"" +
		"                }," +
		"                \"data\": {" +
		"                    \"type\": \"string\"," +
		"                    \"description\": \"Base path for storage\"" +
		"                }," +
		"                \"port\": {" +
		"                    \"type\": \"integer\"," +
		"                    \"description\": \"Port number to bind to\"" +
		"                }," +
		"                \"hostname\": {" +
		"                    \"type\": \"string\"," +
		"                    \"description\": \"The host name where this node will run\"" +
		"                }" +
		"            }" +
		"        }," +
		"        \"aggregator\": {" +
		"            \"type\": \"object\"," +
		"            \"required\": [" +
		"                \"data\"," +
		"                \"port\"" +
		"            ]," +
		"            \"additionalProperties\": false," +
		"            \"properties\": {" +
		"                \"nodeId\": {" +
		"                    \"type\": \"string\"," +
		"                    \"description\": \"The nodeId on the cluster\"" +
		"                }," +
		"                \"port\": {" +
		"                    \"type\": \"integer\"," +
		"                    \"description\": \"Port number to bind to\"" +
		"                }," +
		"                \"data\": {" +
		"                    \"type\": \"string\"," +
		"                    \"description\": \"Base path for storage\"" +
		"                }," +
		"                \"hostname\": {" +
		"                    \"type\": \"string\"," +
		"                    \"description\": \"The host name where this node will run\"" +
		"                }," +
		"                \"leaf_conn_timeout\": {" +
		"                    \"type\": \"integer\"," +
		"                    \"description\": \"Leaf connect timeout, in milliseconds.\"" +
		"                }," +
		"                \"leaf_recv_timeout\": {" +
		"                    \"type\": \"integer\"," +
		"                    \"description\": \"Leaf receive timeout, in milliseconds.\"" +
		"                }," +
		"                \"leaf_send_timeout\": {" +
		"                    \"type\": \"integer\"," +
		"                    \"description\": \"Leaf send timeout, in milliseconds.\"" +
		"                }" +
		"            }" +
		"        }" +
		"    }" +
		"}" +
		""
)

// TODO etcd.go ends up using -2 from the base port
// Make this really independent of that code
func (that *cluster) getEndpointURL(nextHostName *int, nodeHostname string, port int) string {
	var hostname string
	if nodeHostname != "" {
		hostname = nodeHostname
	} else {
		hostname = that.HostNames[*nextHostName]
		*nextHostName = ((*nextHostName) + 1) % len(that.HostNames)
	}
	// FIXME this will break if encryption is enabled
	var returns = "http://" + hostname + ":" + strconv.Itoa(port-2)
	return returns
}

// Preserve the same order as in Etcd::create_initial_urlmap in Cluster.cpp
func (that *cluster) getEndpoints() []string {
	var returns []string
	nextHostName := 0

	// Preserve the same order as in LeafHostInfo::parseClusterConfigNew
	sd := that.StringDictionary
	var endURLStr = that.getEndpointURL(&nextHostName, sd.HostName, sd.Port)
	returns = append(returns, endURLStr)
	for _, item := range that.Leaves {
		var endURLStr = that.getEndpointURL(&nextHostName, item.HostName, item.Port)
		returns = append(returns, endURLStr)
	}
	agg := that.Aggregator
	endURLStr = that.getEndpointURL(&nextHostName, agg.HostName, agg.Port)
	returns = append(returns, endURLStr)

	return returns
}

// TODO Pull the schema into the sources by hardcoding it before the release
func getCluster(clusterFile string) *cluster {
	// schemaLoader := gojsonschema.NewReferenceLoader("file://./cluster.conf.schema.json")
	schemaLoader := gojsonschema.NewStringLoader(clusterSchema)
	documentLoader := gojsonschema.NewReferenceLoader("file://" + clusterFile)

	result, err := gojsonschema.Validate(schemaLoader, documentLoader)
	if err != nil {
		panic(err.Error())
	}

	var clusterConf = cluster{}

	if result.Valid() {
		// Now it is validated parse it
		jsonBytes, err := ioutil.ReadFile(clusterFile)

		if nil == err {
			err = json.Unmarshal(jsonBytes, &clusterConf)
		}
	} else {
		fmt.Printf("The document is not valid. see errors :\n")
		for _, desc := range result.Errors() {
			log.Printf("- %s\n", desc)
		}
	}
	return &clusterConf
}

func getEndpoints(initialPeerUrlsmap string) []string {
	urlMap, _ := types.NewURLsMap(initialPeerUrlsmap)
	returns := []string{}

	for _, v := range urlMap {
		//		fmt.Printf("key[%s] value[%s]\n", k, v)
		var url = v[0]
		urlStr := url.String()
		returns = append(returns, urlStr)
	}

	return returns
}

func getClient(endPoints []string) (*client.Client, error) {
	cfg := client.Config{
		Endpoints:               endPoints,
		Transport:               client.DefaultTransport,
		HeaderTimeoutPerRequest: time.Second,
	}
	var c client.Client
	c, err := client.New(cfg)
	return &c, err
}

func handleCluster(clusterFile string) {
	var cluster = getCluster(clusterFile)
	var endPoints = cluster.getEndpoints()

	c, err := getClient(endPoints)
	if err != nil {
		log.Fatal(err)
	}

	go notifyOnLeader(c, cluster)
}

func isAggregator(leader *client.Member, clust *cluster) bool {
	returns := false

	if clust.Aggregator != nil && clust.Aggregator.NodeID == leader.Name {
		returns = true
	}

	return returns
}

func setPort(in *url.URL, portNum int) {
	host, port, _ := net.SplitHostPort(in.Host)
	port = strconv.Itoa(portNum)
	in.Host = host + ":" + port
}

func getBackendURL(leader *client.Member) (*url.URL, error) {
	// TODO MapDServer.cpp ends up using +1 from the base port
	// Make this really independent of that code
	var returns *url.URL
	var err error

	if len(leader.PeerURLs) > 0 {

		backendURLStr := leader.PeerURLs[0]
		var backendURL *url.URL
		backendURL, err = url.Parse(backendURLStr)
		host, port, err := net.SplitHostPort(backendURL.Host)
		var sPort = ""
		if "" != port && nil == err {
			var iport, _ = strconv.Atoi(port)
			iport = iport + 4
			sPort = ":" + strconv.Itoa(iport)
		}
		scheme := "http"
		if viper.IsSet("ssl-cert") && viper.IsSet("ssl-private-key") {
			scheme = "https"
			http.DefaultTransport.(*http.Transport).TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
		}
		backendURL.Scheme = scheme
		backendURL.Host = host + sPort
		returns = backendURL
	}

	return returns, err
}

func notifyOnLeader(cl *client.Client, clust *cluster) {
	for {
		members := client.NewMembersAPI(*cl)

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		leader, _ := members.Leader(ctx)
		cancel()

		if nil != leader {
			if isAggregator(leader, clust) {
				var err error
				leaderBackendURL, err := getBackendURL(leader)
				if nil == err && leaderBackendURL.String() != backendURL.String() {
					var msg = "Aggregator is changed to " + leader.Name + " at " + leaderBackendURL.String()
					fmt.Println(msg)
					log.Info(msg)

					backendURL = leaderBackendURL
				}
			}
		}

		time.Sleep(200 * time.Millisecond)
	}
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
	var err error
	pflag.IntP("port", "p", 6273, "frontend server port")
	pflag.IntP("http-to-https-redirect-port", "", 6280, "frontend server port for http redirect, when https enabled")
	pflag.StringP("backend-url", "b", "", "url to http-port on omnisci_server [http://localhost:6278]")
	pflag.StringP("cluster_topology", "", "", "The cluster configuration file")
	pflag.StringSliceP("reverse-proxy", "", nil, "additional endpoints to act as reverse proxies, format '/endpoint/:http://target.example.com'")
	pflag.StringP("frontend", "f", "frontend", "path to frontend directory")
	pflag.StringP("jupyter-url", "", "", "url for jupyter integration")
	pflag.StringP("jupyter-prefix", "", "/jupyter", "Jupyter Hub base_url for Jupyter integration")
	pflag.StringP("servers-json", "", "", "path to servers.json")
	pflag.StringP("data", "d", "data", "path to OmniSci data directory")
	pflag.StringP("tmpdir", "", "", "path for temporary file storage [/tmp]")
	pflag.StringP("config", "c", "", "path to OmniSci configuration file")
	pflag.StringP("docs", "", "docs", "path to documentation directory")

	pflag.BoolP("read-only", "r", false, "enable read-only mode")
	pflag.BoolP("quiet", "q", true, "suppress non-error messages")
	pflag.BoolP("verbose", "v", false, "print all log messages to stdout")
	pflag.BoolP("enable-https", "", false, "enable HTTPS support")
	pflag.BoolP("enable-https-authentication", "", false, "enable PKI authentication")
	pflag.BoolP("enable-https-redirect", "", false, "enable HTTP to HTTPS redirect")
	pflag.StringP("cert", "", "cert.pem", "certificate file for HTTPS")
	pflag.StringP("peer-cert", "", "peercert.pem", "peer CA certificate PKI authentication")
	pflag.StringP("ssl-cert", "", "sslcert.pem", "SSL validated public certificate")
	pflag.StringP("ssl-private-key", "", "sslprivate.key", "SSL private key file")
	pflag.StringP("key", "", "key.pem", "key file for HTTPS")
	pflag.DurationP("timeout", "", 60*time.Minute, "maximum request duration")
	pflag.Bool("profile", false, "enable profiling, accessible from /debug/pprof")
	pflag.Bool("compress", false, "enable gzip compression")
	pflag.Bool("metrics", false, "enable Thrift call metrics, accessible from /metrics")
	pflag.Bool("no-strip-omnisci-username-header", false, "do not strip the X-OmniSci-Username header for Jupyter integration")
	pflag.Bool("version", false, "return version")
	pflag.CommandLine.MarkHidden("compress")
	pflag.CommandLine.MarkHidden("profile")
	pflag.CommandLine.MarkHidden("metrics")
	pflag.CommandLine.MarkHidden("quiet")
	pflag.CommandLine.MarkHidden("reverse-proxy")

	pflag.Parse()

	viper.BindPFlag("web.port", pflag.CommandLine.Lookup("port"))
	viper.BindPFlag("web.http-to-https-redirect-port", pflag.CommandLine.Lookup("http-to-https-redirect-port"))
	viper.BindPFlag("web.backend-url", pflag.CommandLine.Lookup("backend-url"))
	viper.BindPFlag("web.reverse-proxy", pflag.CommandLine.Lookup("reverse-proxy"))
	viper.BindPFlag("web.frontend", pflag.CommandLine.Lookup("frontend"))
	viper.BindPFlag("web.jupyter-url", pflag.CommandLine.Lookup("jupyter-url"))
	viper.BindPFlag("web.jupyter-prefix", pflag.CommandLine.Lookup("jupyter-prefix"))
	viper.BindPFlag("web.servers-json", pflag.CommandLine.Lookup("servers-json"))
	viper.BindPFlag("web.enable-https", pflag.CommandLine.Lookup("enable-https"))
	viper.BindPFlag("web.enable-https-authentication", pflag.CommandLine.Lookup("enable-https-authentication"))
	viper.BindPFlag("web.enable-https-redirect", pflag.CommandLine.Lookup("enable-https-redirect"))
	viper.BindPFlag("web.cert", pflag.CommandLine.Lookup("cert"))
	viper.BindPFlag("web.peer-cert", pflag.CommandLine.Lookup("peer-cert"))
	viper.BindPFlag("ssl-cert", pflag.CommandLine.Lookup("ssl-cert"))
	viper.BindPFlag("ssl-private-key", pflag.CommandLine.Lookup("ssl-private-key"))
	viper.BindPFlag("web.key", pflag.CommandLine.Lookup("key"))
	viper.BindPFlag("web.timeout", pflag.CommandLine.Lookup("timeout"))
	viper.BindPFlag("web.profile", pflag.CommandLine.Lookup("profile"))
	viper.BindPFlag("web.compress", pflag.CommandLine.Lookup("compress"))
	viper.BindPFlag("web.metrics", pflag.CommandLine.Lookup("metrics"))
	viper.BindPFlag("web.docs", pflag.CommandLine.Lookup("docs"))
	viper.BindPFlag("web.no-strip-omnisci-username-header", pflag.CommandLine.Lookup("no-strip-omnisci-username-header"))

	viper.BindPFlag("data", pflag.CommandLine.Lookup("data"))
	viper.BindPFlag("tmpdir", pflag.CommandLine.Lookup("tmpdir"))
	viper.BindPFlag("config", pflag.CommandLine.Lookup("config"))
	viper.BindPFlag("read-only", pflag.CommandLine.Lookup("read-only"))
	viper.BindPFlag("quiet", pflag.CommandLine.Lookup("quiet"))
	viper.BindPFlag("verbose", pflag.CommandLine.Lookup("verbose"))
	viper.BindPFlag("version", pflag.CommandLine.Lookup("version"))

	viper.BindPFlag("cluster_topology", pflag.CommandLine.Lookup("cluster_topology"))

	viper.SetDefault("http-port", 6278)

	viper.SetEnvPrefix("MAPD")
	r := strings.NewReplacer(".", "_")
	viper.SetEnvKeyReplacer(r)
	viper.AutomaticEnv()

	viper.SetConfigType("toml")
	viper.AddConfigPath("/etc/mapd")
	viper.AddConfigPath("$HOME/.config/mapd")
	viper.AddConfigPath(".")

	if viper.GetBool("version") {
		fmt.Println("omnisci_web_server " + version)
		os.Exit(0)
	}

	if viper.IsSet("config") {
		viper.SetConfigFile(viper.GetString("config"))
		err := viper.ReadInConfig()
		if err != nil {
			log.Warn("Error reading config file: " + err.Error())
		}
	}

	port = viper.GetInt("web.port")
	httpsRedirectPort = viper.GetInt("web.http-to-https-redirect-port")
	frontend = viper.GetString("web.frontend")
	docsDir = viper.GetString("web.docs")
	serversJSON = viper.GetString("web.servers-json")

	if viper.IsSet("quiet") && !viper.IsSet("verbose") {
		log.Println("Option --quiet is deprecated and has been replaced by --verbose=false, which is enabled by default.")
		verbose = !viper.GetBool("quiet")
	} else {
		verbose = viper.GetBool("verbose")
	}
	dataDir = viper.GetString("data")
	readOnly = viper.GetBool("read-only")
	connTimeout = viper.GetDuration("web.timeout")
	profile = viper.GetBool("web.profile")
	compress = viper.GetBool("web.compress")
	enableMetrics = viper.GetBool("web.metrics")

	backendURLStr := viper.GetString("web.backend-url")
	if backendURLStr == "" {
		s := "http"
		if viper.IsSet("ssl-cert") && viper.IsSet("ssl-private-key") {
			s = "https"
			http.DefaultTransport.(*http.Transport).TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
		}
		backendURLStr = s + "://localhost:" + strconv.Itoa(viper.GetInt("http-port"))
	}

	var cluster = viper.GetString("cluster_topology")
	if cluster != "" {
		handleCluster(cluster)
	}

	backendURL, err = url.Parse(backendURLStr)
	if err != nil {
		log.Fatal(err)
	}

	jupyterURLStr := viper.GetString("web.jupyter-url")
	if jupyterURLStr != "" {
		jupyterURL, err = url.Parse(jupyterURLStr)
		if err != nil {
			log.Fatal(err)
		}
	}

	jupyterPrefix = viper.GetString("web.jupyter-prefix")
	stripOmniSciUsernameHeader = !viper.GetBool("no-strip-omnisci-username-header")

	for _, rp := range viper.GetStringSlice("web.reverse-proxy") {
		s := strings.SplitN(rp, ":", 2)
		if len(s) != 2 {
			log.Fatalln("Could not parse reverse proxy string:", rp)
		}
		path := s[0]
		if len(path) == 0 {
			log.Fatalln("Zero-length path passed for reverse proxy:", rp)
		}
		if path[len(path)-1] != '/' {
			path += "/"
		}
		target, err := url.Parse(s[1])
		if err != nil {
			log.Fatal(err)
		}
		if target.Scheme == "" {
			log.Fatalln("Missing URL scheme, need full URL including http/https:", target)
		}
		proxies = append(proxies, reverseProxy{path, target})
	}

	if os.Getenv("TMPDIR") != "" {
		tmpDir = os.Getenv("TMPDIR")
	}
	if viper.IsSet("tmpdir") {
		tmpDir = viper.GetString("tmpdir")
	}
	if tmpDir != "" {
		err = os.MkdirAll(tmpDir, 0750)
		if err != nil {
			log.Fatal("Could not create temp dir: ", err)
		}
		os.Setenv("TMPDIR", tmpDir)
	}

	enableHTTPS = viper.GetBool("web.enable-https")
	enableHTTPSAuth = viper.GetBool("web.enable-https-authentication")
	enableHTTPSRedirect = viper.GetBool("web.enable-https-redirect")
	certFile = viper.GetString("web.cert")
	keyFile = viper.GetString("web.key")
	peerCertFile = viper.GetString("web.peer-cert")

	registry = metrics.NewRegistry()

	// TODO(andrew): this should be auto-gen'd by Thrift
	thriftMethodMap = make(map[string]thriftMethodTimings)
	thriftMethodMap["render"] = thriftMethodTimings{
		Regex:  regexp.MustCompile(`"?":{"i64":(\d+)`),
		Start:  `"3":{"i64":`,
		Units:  "ms",
		Labels: []string{"execution_time_ms", "render_time_ms", "total_time_ms"},
	}
	thriftMethodMap["sql_execute"] = thriftMethodTimings{
		Regex:  regexp.MustCompile(`"?":{"i64":(\d+)`),
		Start:  `"2":{"i64":`,
		Units:  "ms",
		Labels: []string{"execution_time_ms", "total_time_ms"},
	}

	c := 64
	b := make([]byte, c)
	_, err = rand.Read(b)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	sessionStore = sessions.NewCookieStore(b)
	sessionStore.MaxAge(0)
	serversJSONParams = []string{"username", "password", "database"}
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
		err = errors.New("Uploads disabled: server running in read-only mode")
		return
	}

	uploadDir := dataDir + "/mapd_import/"
	sid := r.Header.Get("sessionid")
	samlAuthCookie, samlAuthCookieErr := r.Cookie(samlAuthCookieName)
	sessionIDCookie, sessionIDCookieErr := r.Cookie(thriftSessionCookieName)
	if samlAuthCookieErr == nil && sessionIDCookieErr == nil && samlAuthCookie.Value == "true" && sessionIDCookie != nil {
		sid = sessionIDCookie.Value
	} else if len(r.FormValue("sessionid")) > 0 {
		sid = r.FormValue("sessionid")
	}

	sessionIDSha256 := sha256.Sum256([]byte(filepath.Base(filepath.Clean(sid))))
	sessionID := hex.EncodeToString(sessionIDSha256[:])
	uploadDir = dataDir + "/mapd_import/" + sessionID + "/"

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
			fn := filepath.Base(filepath.Clean(fh.Filename))
			outfile, err := os.Create(uploadDir + fn)
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

func recordTiming(name string, dur time.Duration) {
	t := registry.GetOrRegister(name, metrics.NewTimer())
	// TODO(andrew): change units to milliseconds if it does not impact other
	// calculations
	t.(metrics.Timer).Update(dur)
}

func recordTimingDuration(name string, then time.Time) {
	dur := time.Since(then)
	recordTiming(name, dur)
}

// ResponseMultiWriter implements an http.ResponseWriter with support for
// outputting to an additional io.Writer.
type ResponseMultiWriter struct {
	io.Writer
	http.ResponseWriter
}

func (w *ResponseMultiWriter) Write(b []byte) (int, error) {
	h := w.ResponseWriter.Header()
	h.Del("Content-Length")
	return w.Writer.Write(b)
}

func hasCustomServersJSONParams(r *http.Request) bool {
	// Checking for form values requires calling ParseForm, which modifies the
	// request buffer and causes issues with the proxy. Solution is to duplicate
	// the request body and reset it after reading.
	b, _ := ioutil.ReadAll(r.Body)
	rdr1 := ioutil.NopCloser(bytes.NewReader(b))
	rdr2 := ioutil.NopCloser(bytes.NewReader(b))
	r.Body = rdr1
	defer func() { r.Body = rdr2 }()
	for _, k := range serversJSONParams {
		if len(r.FormValue(k)) > 0 {
			return true
		}
	}
	return false
}

// thriftTimingHandler records timings for all Thrift method calls. It also
// records timings reported by the backend, as defined by ThriftMethodMap.
// TODO(andrew): use proper Thrift-generated parser
func thriftTimingHandler(h http.Handler) http.Handler {
	return http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" && hasCustomServersJSONParams(r) {
			setServersJSONHandler(rw, r)
			http.Redirect(rw, r, r.URL.Path, http.StatusSeeOther)
			return
		}

		if !enableMetrics || r.Method != "POST" || (r.Method == "POST" && r.URL.Path != "/") {
			h.ServeHTTP(rw, r)
			return
		}

		var thriftMethod string
		body, _ := ioutil.ReadAll(r.Body)
		r.Body = ioutil.NopCloser(bytes.NewReader(body))

		elems := strings.SplitN(string(body), ",", 3)
		if len(elems) > 1 {
			thriftMethod = strings.Trim(elems[1], `"`)
		}

		if len(thriftMethod) < 1 {
			h.ServeHTTP(rw, r)
			return
		}

		tm, exists := thriftMethodMap[thriftMethod]
		defer recordTimingDuration("all", time.Now())
		defer recordTimingDuration(thriftMethod, time.Now())

		if !exists {
			h.ServeHTTP(rw, r)
			return
		}

		buf := new(bytes.Buffer)
		mw := io.MultiWriter(buf, rw)

		rw = &ResponseMultiWriter{
			Writer:         mw,
			ResponseWriter: rw,
		}

		h.ServeHTTP(rw, r)

		go func() {
			offset := strings.LastIndex(buf.String(), tm.Start)
			if offset >= 0 {
				timings := tm.Regex.FindAllStringSubmatch(buf.String()[offset:], len(tm.Labels))
				for k, v := range timings {
					dur, _ := time.ParseDuration(v[1] + tm.Units)
					recordTiming(thriftMethod+"."+tm.Labels[k], dur)
				}
			}
		}()
	})
}

func metricsHandler(rw http.ResponseWriter, r *http.Request) {
	if len(r.FormValue("enable")) > 0 {
		enableMetrics = true
	} else if len(r.FormValue("disable")) > 0 {
		enableMetrics = false
	}
	jsonBuf := new(bytes.Buffer)
	metrics.WriteJSONOnce(registry, jsonBuf)
	ijsonBuf := new(bytes.Buffer)
	json.Indent(ijsonBuf, jsonBuf.Bytes(), "", "  ")
	rw.Write(ijsonBuf.Bytes())
}

func metricsResetHandler(rw http.ResponseWriter, r *http.Request) {
	registry.UnregisterAll()
	metricsHandler(rw, r)
}

func setServersJSONHandler(rw http.ResponseWriter, r *http.Request) {
	session, _ := sessionStore.Get(r, "servers-json")

	for _, key := range serversJSONParams {
		if len(r.FormValue(key)) > 0 {
			session.Values[key] = r.FormValue(key)
		}
	}

	session.Save(r, rw)
}

func clearServersJSONHandler(rw http.ResponseWriter, r *http.Request) {
	session, _ := sessionStore.Get(r, "servers-json")

	session.Options.MaxAge = -1

	session.Save(r, rw)
}

func docsHandler(rw http.ResponseWriter, r *http.Request) {
	h := http.StripPrefix("/docs/", http.FileServer(http.Dir(docsDir)))
	h.ServeHTTP(rw, r)
}

// samlPostHandler receives a XML SAML payload from a provider (e.g. Okta) and
// then makes a connect call to OmniSciDB with the base64'd payload. If the call succeeds
// we then set a session cookie (`omnisci_session`) for Immerse to use for login, as well
// as the username (`omnisci_username`) and db name (`omnisci_db`).
func samlPostHandler(rw http.ResponseWriter, r *http.Request) {
	var err error
	ok := false
	targetPage := "/"

	if r.Method == "POST" {
		var sessionToken string

		b64ResponseXML := r.FormValue("SAMLResponse")

		// This is what a Thrift connect call to OmniSciDB looks like. Here, the username and database
		// name are left blank, per SAML login conventions. Hand-crafting Thrift messages like this
		// isn't exactly "best practices", but it beats importing a whole Thrift lib for just this.
		var jsonString = []byte(`[1,"connect",1,0,{"2":{"str":"` + b64ResponseXML + `"},"3":{"str":""}}]`)

		resp, err := http.Post(backendURL.String(), "application/vnd.apache.thrift.json", bytes.NewBuffer(jsonString))
		if err != nil {
			return
		}

		bodyBytes, _ := ioutil.ReadAll(resp.Body)
		resp.Body.Close()

		jsonParsed, _ := gabs.ParseJSON(bodyBytes)
		if err != nil {
			return
		}

		relayState := r.FormValue("RelayState")
		if relayState != "" {
			targetPage = relayState
		}

		// We should have one of the two following payloads at this point:
		// 		Success => [1,"connect",2,0,{"0":{"str":"5h6KW9NTv1ef1kOfOlAGN9q63usKOg0i"}}]
		// 		Failure => [1,"connect",2,0,{"1":{"rec":{"1":{"str":"Invalid credentials."}}}}]
		// Only set the cookie if we can parse a success payload.
		sessionToken, ok = jsonParsed.Index(4).Search("0", "str").Data().(string)
		if ok {
			sessionIDCookie := http.Cookie{
				Name:     thriftSessionCookieName,
				Value:    sessionToken,
				HttpOnly: true,
			}
			http.SetCookie(rw, &sessionIDCookie)

			samlFlagCookie := http.Cookie{
				Name:  samlAuthCookieName,
				Value: "true",
			}
			http.SetCookie(rw, &samlFlagCookie)
		}
	}

	defer func() {
		if ok {
			http.Redirect(rw, r, targetPage, 301)
		} else {
			var errorString string
			if err != nil {
				errorString = err.Error()
			} else {
				errorString = "invalid credentials"
			}
			http.Redirect(rw, r, samlErrorPage, 303)
			log.Infoln("Error logging user in via SAML: ", errorString)
		}
	}()
}

type ServeIndexOn404FileSystem struct {
	http.FileSystem
	Filename string
}

func (fs ServeIndexOn404FileSystem) Open(name string) (http.File, error) {
	file, err := fs.FileSystem.Open(name)
	if os.IsNotExist(err) {
		if strings.HasPrefix(name, "/beta/") {
			file, err = fs.FileSystem.Open("/beta/index.html")
		} else {
			file, err = fs.FileSystem.Open("/index.html")
		}
	}

	if err == nil {
		if stat, statErr := file.Stat(); statErr != nil {
			fs.Filename = stat.Name()
		}
	}

	return file, err
}

func thriftOrFrontendHandler(rw http.ResponseWriter, r *http.Request) {
	fs := ServeIndexOn404FileSystem{http.Dir(frontend), ""}
	h := http.StripPrefix("/", http.FileServer(fs))

	if r.Method == "POST" {
		h = httputil.NewSingleHostReverseProxy(backendURL)
		rw.Header().Del("Access-Control-Allow-Origin")

		// If the thriftSessionCookieName is present, it holds the real session ID, while the Thrift
		// call is using a placeholder. This code replaces the fake session ID in the Thrift call
		// with the real one from the cookie.
		samlAuthCookie, samlAuthCookieErr := r.Cookie(samlAuthCookieName)
		sessionIDCookie, sessionIDCookieErr := r.Cookie(thriftSessionCookieName)
		if samlAuthCookieErr == nil && sessionIDCookieErr == nil && samlAuthCookie.Value == "true" && sessionIDCookie != nil {
			bodyBytes, _ := ioutil.ReadAll(r.Body)
			defer r.Body.Close()

			// In general, if we encounter any errors, we want to make this session code a noop
			jsonParsed, err := gabs.ParseJSON(bodyBytes)
			if err == nil {
				// Grab the session ID from the thrift call
				sessionToken, ok := jsonParsed.Index(4).Search("1", "str").Data().(string)

				// If the session ID is our known placeholder ID, replace it with the real one
				if ok && sessionToken == samlPlaceholderSessionID {
					jsonParsed.Index(4).Set(sessionIDCookie.Value, "1", "str")

					r.Body = ioutil.NopCloser(bytes.NewReader([]byte(jsonParsed.String())))
					r.ContentLength = int64(len(jsonParsed.String()))
				} else {
					r.Body = ioutil.NopCloser(bytes.NewReader(bodyBytes))
				}
			} else {
				r.Body = ioutil.NopCloser(bytes.NewReader(bodyBytes))
			}
		}
	}

	if r.Method == "GET" && (r.URL.Path == "/" || r.URL.Path == "/beta/" || strings.HasSuffix(fs.Filename, ".html")) {
		rw.Header().Del("Cache-Control")
		rw.Header().Add("Cache-Control", "no-cache, no-store, must-revalidate")
	}

	h.ServeHTTP(rw, r)
}

func betaOrRedirectFrontendHandler(rw http.ResponseWriter, r *http.Request) {
	cookie, err := r.Cookie("omnisci-beta")
	if err != nil || cookie.Value != "true" {
		http.Redirect(rw, r, "/", http.StatusTemporaryRedirect)
		return
	}

	thriftOrFrontendHandler(rw, r)
}

// Retrieve the actual OmniSci Thrift session ID, accounting for each possible source
func getSessionIDForJupyter(r *http.Request) string {
	samlAuthCookie, samlAuthCookieErr := r.Cookie(samlAuthCookieName)
	sessionIDCookie, sessionIDCookieErr := r.Cookie(thriftSessionCookieName)

	if samlAuthCookieErr == nil && sessionIDCookieErr == nil && samlAuthCookie.Value == "true" && sessionIDCookie != nil {
		// Session was authenticated using SAML, session ID stored on secure cookie
		return sessionIDCookie.Value
	}

	// Session was authenticated normally, session ID stored in client session cookie
	clientSessionCookie, clientSessionCookieErr := r.Cookie(clientSessionCookieName)

	if clientSessionCookieErr == nil && clientSessionCookie != nil {
		return clientSessionCookie.Value
	}

	return ""
}

// Function copied from https://golang.org/src/net/http/httputil/reverseproxy.go source
func cloneHeader(h http.Header) http.Header {
	h2 := make(http.Header, len(h))
	for k, vv := range h {
		vv2 := make([]string, len(vv))
		copy(vv2, vv)
		h2[k] = vv2
	}
	return h2
}

// Function partially from https://golang.org/src/net/http/httputil/reverseproxy.go source
func cloneRequest(r *http.Request) *http.Request {
	outreq := r.WithContext(r.Context())
	if r.ContentLength == 0 {
		outreq.Body = nil
	}

	outreq.Header = cloneHeader(r.Header)

	return outreq
}

func cloneJupyterRequest(r *http.Request, sessionID string, username string) *http.Request {
	// Create a copy of the incoming request and point it towards Jupyter
	newReq := cloneRequest(r)
	newReq.URL.Scheme = jupyterURL.Scheme
	newReq.URL.Host = jupyterURL.Host

	return newReq
}

type jupyterNotebookParams struct {
	NotebookName string
	SQL          string
}

// Double-escape any double quotes in SQL, since it will be placed
// inside both the Ibis string value in the notebook as well as
// the JSON string value of the notebook JSON document
func (p jupyterNotebookParams) EscapedSQL() string {
	json, err := json.Marshal(p.SQL)

	if err != nil {
		log.Fatalln("Errors parsing SQL input to Jupyter notebook:", err)
	}

	jsonString := string(json)
	innerJSONString := jsonString[1 : len(jsonString)-1]
	escapedInnerJSONString := strings.Replace(innerJSONString, `\"`, `\\\"`, -1)

	return escapedInnerJSONString
}

// Time format, not a hardcoded string - See https://golang.org/src/time/format.go
var jupyterNotebookNameFormat = "OmniSci_2006-01-02_15:04:05.ipynb"

// Text template parsed and constructed for use in main() below
var jupyterNotebookTemplate *template.Template
var jupyterNotebookTemplateText = `{
	"name": "{{.NotebookName}}",
	"path": "{{.NotebookName}}",
	"type": "notebook",
	"format": "json",
	"content": {
		"cells": [
			{
				"cell_type": "code",
				"execution_count": null,
				"metadata": {
					"trusted": true
				},
				"outputs": [],
				"source": [
					"# An Ibis connection object (con) is created on notebook startup, which\n",
					"# includes a pymapd connection object as a property (con.con).\n",
					"# If you receive a session invalid or object not found error using it,\n",
					"# please close the Jupyter Lab browser tab, relaunch from Immerse,\n",
					"# and run this cell to recreate your con object using the\n",
					"# omnisci_connect function.\n",
					"con = omnisci_connect()\n",
					{{ if .SQL -}}
						"o = con.sql(\"\"\"{{ .EscapedSQL }}\"\"\")\n",
						"o"
					{{- else -}}
						"con.list_tables()"
					{{- end }}
				]
			}
		],
		"metadata": {
			"kernelspec": {
				"display_name": "Python 3",
				"language": "python",
				"name": "python3"
			},
			"language_info": {
				"codemirror_mode": {
					"name": "ipython",
					"version": 3
				},
				"file_extension": ".py",
				"mimetype": "text/x-python",
				"name": "python",
				"nbconvert_exporter": "python",
				"pygments_lexer": "ipython3",
				"version": "3.7.3"
			}
		},
		"nbformat": 4,
		"nbformat_minor": 4
	}
}`

func checkForJupyterError(r *http.Response) error {
	if r.StatusCode >= 200 && r.StatusCode < 300 {
		return nil
	}

	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		return errors.New(r.Status)
	}

	json, err := gabs.ParseJSON(body)
	if err != nil {
		return errors.New(r.Status)
	}

	msg, ok := json.Path("message").Data().(string)
	if ok {
		return errors.New(msg)
	}
	return errors.New(r.Status)
}

func createNewNotebook(r *http.Request, sessionID string, username string, sql string) error {
	notebookName := time.Now().Format(jupyterNotebookNameFormat)

	// Create new notebook
	notebookParams := jupyterNotebookParams{
		NotebookName: notebookName,
		SQL:          sql,
	}

	var notebookBuffer bytes.Buffer
	jupyterNotebookTemplate.Execute(&notebookBuffer, notebookParams)

	// For this and following requests, use the incoming request to clone from and reuse auth headers
	notebookCreateReq := cloneJupyterRequest(r, sessionID, username)
	notebookCreateReq.Method = http.MethodPut
	notebookCreateReq.URL.Path = jupyterPrefix + "/user/" + username + "/api/contents/" + notebookName
	notebookCreateReq.Body = ioutil.NopCloser(&notebookBuffer)

	notebookCreateResp, err := http.DefaultTransport.RoundTrip(notebookCreateReq)
	if err != nil {
		return err
	}
	defer notebookCreateResp.Body.Close()

	err = checkForJupyterError(notebookCreateResp)
	if err != nil {
		return err
	}

	// Get the Jupyter workspace definition

	// The original request is already to what we want - just clone and send it
	// Later on, the original will go through the main reverse proxy and get the updated one we put
	getWorkspaceRequest := cloneJupyterRequest(r, sessionID, username)

	workspaceResp, err := http.DefaultTransport.RoundTrip(getWorkspaceRequest)
	if err != nil {
		return err
	}
	defer workspaceResp.Body.Close()

	if workspaceResp.StatusCode > 299 {
		return errors.New("Getting workspace definition failed with status " + workspaceResp.Status)
	}

	workspaceBodyBytes, err := ioutil.ReadAll(workspaceResp.Body)
	if err != nil {
		return err
	}

	workspace, err := gabs.ParseJSON(workspaceBodyBytes)
	if err != nil {
		return err
	}

	// Modify workspace to have our new notebook open in a tab and focused
	notebookRef := "notebook:" + notebookName
	workspace.ArrayAppend(notebookRef, "data", "layout-restorer:data", "main", "dock", "widgets")
	workspace.Set(notebookRef, "data", "layout-restorer:data", "main", "current")
	workspace.Set(notebookName, "data", notebookRef, "data", "path")
	workspace.Set("Notebook", "data", notebookRef, "data", "factory")

	workspaceString := workspace.String()

	// Now put our modified workspace back again
	putWorkspaceRequest := cloneJupyterRequest(r, sessionID, username)
	putWorkspaceRequest.Method = http.MethodPut
	putWorkspaceRequest.Header["Content-Type"] = []string{"application/json;charset=utf-8"}
	putWorkspaceRequest.Header["Content-Length"] = []string{strconv.Itoa(len(workspaceString))}
	putWorkspaceRequest.ContentLength = int64(len(workspaceString))
	putWorkspaceRequest.Body = ioutil.NopCloser(strings.NewReader(workspaceString))

	putWorkspaceResp, err := http.DefaultTransport.RoundTrip(putWorkspaceRequest)
	if err != nil {
		return err
	}
	defer putWorkspaceResp.Body.Close()

	err = checkForJupyterError(putWorkspaceResp)
	if err != nil {
		return err
	}

	return nil
}

type jupyterSessionFileParams struct {
	SessionID string
}

var jupyterSessionFileTemplate *template.Template
var jupyterSessionFileTemplateText = `{
	"name": ".omniscisession",
	"path": ".omniscisession",
	"type": "file",
	"format": "text",
	"content": "{\"session\": \"{{.SessionID}}\"}"
}`

func createJupyterSessionFile(r *http.Request, sessionID, username string) error {
	sessionParams := jupyterSessionFileParams{
		SessionID: sessionID,
	}

	var sessionBuffer bytes.Buffer
	jupyterSessionFileTemplate.Execute(&sessionBuffer, sessionParams)

	// clone incoming request to reuse auth headers
	sessionCreateReq := cloneJupyterRequest(r, sessionID, username)
	sessionCreateReq.Method = http.MethodPut
	sessionCreateReq.URL.Path = jupyterPrefix + "/user/" + username + "/api/contents/.jupyterscratch/.omniscisession"
	sessionCreateReq.Header["Content-Length"] = []string{strconv.Itoa(sessionBuffer.Len())}
	sessionCreateReq.ContentLength = int64(sessionBuffer.Len())
	sessionCreateReq.Body = ioutil.NopCloser(&sessionBuffer)

	sessionCreateResp, err := http.DefaultTransport.RoundTrip(sessionCreateReq)
	if err != nil {
		return err
	}
	defer sessionCreateResp.Body.Close()

	err = checkForJupyterError(sessionCreateResp)
	if err != nil {
		return err
	}

	return nil
}

func handleJupyterError(rw http.ResponseWriter, msg string, err error) {
	if err != nil {
		msg += ": " + err.Error()
	}

	rw.WriteHeader(500)
	rw.Write([]byte(msg))
	log.Println(msg)
}

func jupyterProxyHandler(rw http.ResponseWriter, r *http.Request) {
	h := httputil.NewSingleHostReverseProxy(jupyterURL)

	pathParts := strings.Split(r.URL.Path, "/")

	// Match iff we are hitting GET /jupyter/hub/login,
	// where we need to receive query params and pass session ID header
	if len(pathParts) > 3 && pathParts[2] == "hub" && pathParts[3] == "login" {
		// Set session ID header for the Jupyter Hub OmniSci authenticator
		sessionID := getSessionIDForJupyter(r)
		if sessionID == "" {
			handleJupyterError(rw, "Failed to get session ID from request", nil)
			return
		}
		r.Header["X-OmniSci-SessionID"] = []string{sessionID}
		if stripOmniSciUsernameHeader {
			r.Header.Del("X-OmniSci-Username")
		}

		queryValues, err := url.ParseQuery(r.URL.RawQuery)

		newnotebook := queryValues["newnotebook"]
		if err == nil && newnotebook != nil && newnotebook[0] == "true" {
			// Set cookies on the client indicating it wants to make a new OmniSci notebook on login
			newNotebookCookie := &http.Cookie{
				Name:     "newnotebook",
				Value:    "true",
				Path:     "/",
				Expires:  time.Now().Add(5 * time.Minute),
				HttpOnly: true,
			}
			http.SetCookie(rw, newNotebookCookie)

			sqlValues := queryValues["sql"]
			if sqlValues != nil {
				escapedSQL := url.QueryEscape(sqlValues[0])

				newNotebookSQLCookie := &http.Cookie{
					Name:     "newnotebooksql",
					Value:    escapedSQL,
					Path:     "/",
					Expires:  time.Now().Add(5 * time.Minute),
					HttpOnly: true,
				}
				http.SetCookie(rw, newNotebookSQLCookie)
			}
		}
	} else if len(pathParts) > 7 && pathParts[4] == "lab" && pathParts[5] == "api" && pathParts[6] == "workspaces" && pathParts[7] == "lab" {
		// Match iff we are hitting GET /jupyter/user/<username>/lab/api/workspaces/lab for the first time -
		// This is the request Hub sends to get the user's visible workspace (open notebooks)

		// Retrieve the cookies we set earlier
		newNotebookCookie, err := r.Cookie("newnotebook")
		if err == nil && newNotebookCookie.Value == "true" {
			newNotebookClearCookie := &http.Cookie{
				Name:     "newnotebook",
				Value:    "",
				Path:     "/",
				Expires:  time.Unix(0, 0),
				MaxAge:   -1,
				HttpOnly: true,
			}
			http.SetCookie(rw, newNotebookClearCookie)

			newNotebookSQLCookie, err := r.Cookie("newnotebooksql")
			sql := ""
			if err == nil {
				sql, err = url.QueryUnescape(newNotebookSQLCookie.Value)

				if err != nil {
					log.Fatal("Error unescaping SQL cookie for Jupyter: ", err)
				}

				newNotebookSQLClearCookie := &http.Cookie{
					Name:     "newnotebooksql",
					Value:    "",
					Path:     "/",
					Expires:  time.Unix(0, 0),
					MaxAge:   -1,
					HttpOnly: true,
				}
				http.SetCookie(rw, newNotebookSQLClearCookie)
			}

			// Create new notebook
			sessionID := getSessionIDForJupyter(r)
			if sessionID == "" {
				handleJupyterError(rw, "Failed to get session ID from request", nil)
				return
			}
			username := pathParts[3]

			// create session file
			createErr := createJupyterSessionFile(r, sessionID, username)
			if createErr != nil {
				handleJupyterError(rw, "Error creating session file", createErr)
				return
			}

			// create new notebook
			createErr = createNewNotebook(r, sessionID, username, sql)
			if createErr != nil {
				handleJupyterError(rw, "Error creating notebook", createErr)
				return
			}
		}
	}

	// Pass original request through and reverse proxy
	h.ServeHTTP(rw, r)
}

func httpToHTTPSRedirectHandler(rw http.ResponseWriter, r *http.Request) {
	// Redirect HTTP request to same URL with only two changes: https scheme,
	// and the main server port configured in the 'port' param, rather than the
	// incoming port ('http-to-https-redirect-port')
	requestHost, _, _ := net.SplitHostPort(r.Host)
	redirectURL := url.URL{Scheme: "https", Host: requestHost + ":" + strconv.Itoa(port), Path: r.URL.Path, RawQuery: r.URL.RawQuery}
	http.Redirect(rw, r, redirectURL.String(), http.StatusTemporaryRedirect)
}

func (rp *reverseProxy) proxyHandler(rw http.ResponseWriter, r *http.Request) {
	h := http.StripPrefix(rp.Path, httputil.NewSingleHostReverseProxy(rp.Target))
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

func modifyServersJSON(r *http.Request, orig []byte) ([]byte, error) {
	session, _ := sessionStore.Get(r, "servers-json")
	j, err := gabs.ParseJSON(orig)
	if err != nil {
		return nil, err
	}

	jj, err := j.Children()
	if err != nil {
		return nil, err
	}

	for _, key := range serversJSONParams {
		if session.Values[key] != nil {
			_, err = jj[0].Set(session.Values[key].(string), key)
			if err != nil {
				return nil, err
			}
		}
	}

	return j.BytesIndent("", "  "), nil
}

func serversHandler(rw http.ResponseWriter, r *http.Request) {
	var j []byte
	servers := ""
	subDir := filepath.Dir(r.URL.Path)
	if len(serversJSON) > 0 {
		servers = serversJSON
	} else {
		servers = frontend + subDir + "/servers.json"
		if _, err := os.Stat(servers); os.IsNotExist(err) {
			servers = frontend + "/servers.json"
		}
	}
	j, err := ioutil.ReadFile(servers)
	if err != nil {
		s := server{}
		s.Master = true
		s.Username = "admin"
		s.Password = "HyperInteractive"
		s.Database = "omnisci"
		h, p, _ := net.SplitHostPort(r.Host)
		s.Port, _ = net.LookupPort("tcp", p)
		s.Host = h
		// handle IPv6 addresses
		ip := net.ParseIP(h)
		if ip != nil && ip.To4() == nil {
			s.Host = "[" + h + "]"
		}

		ss := []server{s}
		j, _ = json.Marshal(ss)
	}

	jj, err := modifyServersJSON(r, j)
	if err != nil {
		msg := "Error processing servers.json: " + err.Error()
		http.Error(rw, msg, http.StatusInternalServerError)
		log.Println(msg)
		return
	}

	rw.Header().Del("Cache-Control")
	rw.Header().Add("Cache-Control", "no-cache, no-store, must-revalidate")
	rw.Write(jj)
}

func versionHandler(rw http.ResponseWriter, r *http.Request) {
	outVers := "OmniSciDB:\n" + version
	versTxt := frontend + "/version.txt"
	feVers, err := ioutil.ReadFile(versTxt)
	if err == nil {
		outVers += "\n\n"
		outVers += "Immerse:\n"
		outVers += string(feVers)
	}
	rw.Write([]byte(outVers))
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
	if !verbose {
		log.SetOutput(lf)
		log.SetFormatter(&log.TextFormatter{
			DisableColors: true,
			FullTimestamp: true,
		})

		alog = alf
	} else {
		log.SetOutput(io.MultiWriter(os.Stdout, lf))
		alog = io.MultiWriter(os.Stdout, alf)
	}

	jupyterNotebookTemplate, err = template.New("jupyter-notebook").Parse(jupyterNotebookTemplateText)
	if err != nil {
		log.Fatalln("Error parsing Jupyter notebook template: ", err)
	}

	jupyterSessionFileTemplate, err = template.New("jupyter-session").Parse(jupyterSessionFileTemplateText)
	if err != nil {
		log.Fatalln("Error parsing Jupyter session file template: ", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/saml-post", samlPostHandler)
	mux.HandleFunc("/upload", uploadHandler)
	mux.HandleFunc("/downloads/", downloadsHandler)
	mux.HandleFunc("/deleteUpload", deleteUploadHandler)
	mux.HandleFunc("/servers.json", serversHandler)
	mux.HandleFunc("/", thriftOrFrontendHandler)
	mux.HandleFunc("/beta/", betaOrRedirectFrontendHandler)
	mux.HandleFunc("/docs/", docsHandler)
	mux.HandleFunc("/metrics/", metricsHandler)
	mux.HandleFunc("/metrics/reset/", metricsResetHandler)
	mux.HandleFunc("/version.txt", versionHandler)
	mux.HandleFunc("/_internal/set-servers-json", setServersJSONHandler)
	mux.HandleFunc("/_internal/clear-servers-json", clearServersJSONHandler)

	if jupyterURL != nil {
		mux.HandleFunc(jupyterPrefix+"/", jupyterProxyHandler)
	}

	if profile {
		mux.HandleFunc("/debug/pprof/", pprof.Index)
		mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
		mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
		mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	}

	for k := range proxies {
		rp := proxies[k]
		log.Infoln("Proxy:", rp.Path, "to", rp.Target)
		mux.HandleFunc(rp.Path, rp.proxyHandler)
	}

	c := cors.New(cors.Options{
		AllowedHeaders: []string{"Accept", "Cache-Control", "Content-Type", "sessionid", "X-Requested-With"},
	})
	cmux := c.Handler(mux)
	cmux = handlers.LoggingHandler(alog, cmux)
	cmux = thriftTimingHandler(cmux)
	if compress {
		cmux = handlers.CompressHandler(cmux)
	}

	tlsConfig := &tls.Config{}
	if enableHTTPSAuth {
		caCert, err := ioutil.ReadFile(peerCertFile)
		if err != nil {
			log.Fatalln("Errors opening peer file:", err, peerCertFile)
		}
		caCertPool := x509.NewCertPool()
		caCertPool.AppendCertsFromPEM(caCert)
		tlsConfig = &tls.Config{
			ClientCAs:  caCertPool,
			ClientAuth: tls.RequireAndVerifyClientCert,
		}
		tlsConfig.BuildNameToCertificate()

	}

	srv := &graceful.Server{
		Timeout: 5 * time.Second,
		Server: &http.Server{
			Addr:         ":" + strconv.Itoa(port),
			Handler:      cmux,
			ReadTimeout:  connTimeout,
			WriteTimeout: connTimeout,
			TLSConfig:    tlsConfig,
		},
	}

	if enableHTTPS {
		if _, err := os.Stat(certFile); err != nil {
			log.Fatalln("Error opening certificate:", err)
		}
		if _, err := os.Stat(keyFile); err != nil {
			log.Fatalln("Error opening keyfile:", err)
		}

		if enableHTTPSRedirect {
			go func() {
				err := http.ListenAndServe(":"+strconv.Itoa(httpsRedirectPort), http.HandlerFunc(httpToHTTPSRedirectHandler))

				if err != nil {
					log.Fatalln("Error starting http redirect listener:", err)
				}
			}()
		}

		err = srv.ListenAndServeTLS(certFile, keyFile)
	} else {
		err = srv.ListenAndServe()
	}

	if err != nil {
		log.Fatal("Error starting http server: ", err)
	}
}
