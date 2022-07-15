docker build -f Dockerfile -t hdk
docker run  --privileged -it --name hdk -v /localdisk/pakurapo/omniscidb:/omniscidb hdk:latest