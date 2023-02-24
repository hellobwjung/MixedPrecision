export http_proxy="http://12.26.204.100:8080"
export https_proxy="https://12.26.204.100:8080"
export no_proxy="12.*.*.*,*.samsung.net,127.0.0.1,pfs.nprotect.com,127.0.0.1:16105,127.0.0.1:16106,127.0.0.1:16108,127.0.0.1:16107,*.dsvdi.net,.samsungds.net,localhost,127.0.0.1:14440"
export ftp_proxy="ftp://12.26.204.100:8080"


export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$http_proxy
export NO_PROXY=$no_proxy

export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
