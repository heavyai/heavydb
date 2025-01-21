ARG from

# Download the official headers from github.com/KhronosGroup
FROM rockylinux:8 as khronos

RUN dnf install -y \
        git && \
    rm -rf /var/cache/dnf/*

RUN git clone https://github.com/KhronosGroup/OpenGL-Registry.git && cd OpenGL-Registry && \
    git checkout 681c365c012ac9d3bcadd67de10af4730eb460e0 && \
    cp -r api/GL /usr/local/include

RUN git clone https://github.com/KhronosGroup/EGL-Registry.git && cd EGL-Registry && \
    git checkout 0fa0d37da846998aa838ed2b784a340c28dadff3 && \
    cp -r api/EGL api/KHR /usr/local/include

RUN git clone --branch=mesa-17.3.3 --depth=1 https://gitlab.freedesktop.org/mesa/mesa.git && cd mesa && \
    cp include/GL/gl.h include/GL/gl_mangle.h /usr/local/include/GL/

FROM ${from}

RUN dnf install -y 'dnf-command(config-manager)'

RUN dnf config-manager --set-enabled powertools

RUN dnf install -y \
        pkgconfig \
        libXau-devel \
        libXdmcp-devel \
        libxcb-devel \
        libXext-devel \
        libX11-devel && \
    rm -rf /var/cache/dnf/*

COPY --from=khronos /usr/local/include /usr/local/include

ENV PKG_CONFIG_PATH /usr/local/lib64/pkgconfig:/usr/local/lib/pkgconfig

COPY usr /usr
