MapD documentation, written in reStructured Text and compiled by Sphinx.

Requires pip, virtualenv, and pdflatex. Instructions for installing these
dependencies are below.

# ReStructured Text

RST primers:

    http://docutils.sourceforge.net/docs/user/rst/quickstart.html
    http://docutils.sourceforge.net/docs/user/rst/quickref.html
    http://www.sphinx-doc.org/en/stable/rest.html

# Organization

Raw documentation resides in .rst files under a directory named for each guide.
Currently:

    immerse-user-guide/
    mapd-guide/
    release-notes/

To add a new section, list it in the `index.rst` for the relevant guide.

## Handling Images

Images should be selected with care: remember that PDFs are expected to be
static, possibly printed, documents. Therefore if adding a .gif, also provide a
static version of it as a .jpg. In your document refer to these images with
`base.*` instead of `base.gif`, and the Sphinx will select an acceptable file
type for PDF and HTML.

For historical reasons .png is preferred over .jpg and .gif. In other words if
you provide three files named `scatter.png`, `scatter.gif`, and `scatter.jpg`
and refer to the images as `scatter.*`, Sphinx will first try to use
`scatter.png` if it exists, then will try `scatter.gif`, and finally will try
`scatter.jpg`, in that order.

This is why images in the Immerse User Guide have unique base names. For
example, you should reference the heatmap creation images as `heatmap-create.*`
and provide a .gif named `heatmap-create.gif` as well as the static
`heatmap-create.jpg` (which will be used for document types which do not
support .gif - PDFs). The example heatmap image has a unique base name of
`heatmap-header` so that that that file (`heatmap-header.png`) is not
picked up when referencing `heatmap-create.*`.

## Theme

HTML/CSS theme files are in `_themes/mapd_docs_theme`.

# Build and Preview

To view changes, rebuild the documentation with:

    ./build.sh

HTML will be in `build/dirhtml`. Individual PDFs will be in `build/latex`.

If doing a full mapd build managed by CMake, you can instead do:

    make docs

and view them in `$CMAKE_BUILD_DIR/docs/`.

You may wish to run a web server so that your browser correctly loads
`index.html` when visiting a new directory. Run one of the following and then
visit http://localhost:8282 :

    python -m SimpleHTTPServer 8282
    python3 -m http.server 8282

# Installing Dependencies

CentOS:

    sudo yum install python-pip python-virtualenv
    sudo yum install texlive texlive-latex-bin-bin "texlive-*"

Ubuntu:

    sudo apt install python-pip virtualenv
    sudo apt install texlive-latex-base texlive-full

MacOS:

    sudo easy_install pip
    sudo pip install virtualenv
    brew cask install mactex # or follow instructions at
                             # https://www.tug.org/mactex/
