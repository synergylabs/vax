# To build:
#   docker build --tag ytyou/ticktock:0.1.3-alpha --tag ytyou/ticktock:latest
#     --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
#     --build-arg GIT_COMMIT=$(git log -1 --pretty=format:%h)
#     --build-arg VERSION=0.1.3-alpha --no-cache=true --add-host=ticktock
#
# To inspect the image by running it:
#   docker run --rm -it --entrypoint=/bin/bash ytyou/ticktock:latest
#
# To update the image, first run it:
#   docker run -it --entrypoint=/bin/bash ytyou/ticktock:latest
# While inside the container, update anything you want. For example,
# you can update the /opt/ticktock/conf/ticktock.conf file. After you
# are done updating, exit the container and run:
#   docker commit <container-id> <new_image_name>
#
# To start ticktock container for the first time:
#   docker run -d --name ticktock -p 6181-6182:6181-6182 -p 6181:6181/udp ytyou/ticktock:latest
#
# To attach to a running ticktock container:
#   docker exec -it ticktock /bin/bash
#
# To stop a running ticktock container:
#   docker stop ticktock
#
# To restart ticktock container:
#   docker start ticktock

FROM ubuntu

ARG BUILD_DATE
ARG GIT_COMMIT
ARG VERSION
ARG DEBIAN_FRONTEND=noninteractive

LABEL name="TickTock" version="$VERSION"
LABEL build-date="$BUILD_DATE" git-commit="$GIT_COMMIT"
LABEL maintainer="yongtao.you@gmail.com"
LABEL url="https://github.com/ytyou/ticktock"
LABEL description="An OpenTSDB-like time series database, with much better performance."
LABEL docker.cmd="docker run -d --name ticktock -p 6181-6182:6181-6182 -p 6181:6181/udp ytyou/ticktock:latest"

HEALTHCHECK --interval=5m --timeout=5s \
  CMD /opt/ticktock/scripts/healthcheck.sh
STOPSIGNAL SIGINT

RUN apt-get update && apt-get install -y \
  curl \
  nano

RUN useradd -M -U ticktock

RUN mkdir -p /var/lib/ticktock && \
  chown ticktock:ticktock /var/lib/ticktock
RUN mkdir -p /opt/ticktock && \
  chown ticktock:ticktock /opt/ticktock

COPY limits.conf /etc/security/
COPY --chown=ticktock:ticktock opt/ /opt/

EXPOSE 6181/tcp
EXPOSE 6181/udp
EXPOSE 6182/tcp

USER ticktock
WORKDIR /tmp
VOLUME /var/lib/ticktock
ENTRYPOINT ["/opt/ticktock/scripts/entrypoint.sh"]
