FROM ubuntu:22.04
LABEL maintainer="HEAVY.AI Support <support@heavy.ai>"

WORKDIR /opt/heavyai

RUN mkdir bin

COPY license_server /opt/heavyai/bin

EXPOSE 6280

CMD /opt/heavyai/bin/license_server --storage /var/lib/heavyai/storage --config /var/lib/heavyai/license_server.conf
