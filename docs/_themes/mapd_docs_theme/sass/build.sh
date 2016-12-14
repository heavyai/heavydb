#!/bin/bash

if ! hash sass 2>/dev/null ; then
  echo "Please run 'gem install sass' and ensure 'sass' is in your PATH"
  exit
fi

rm -f ../static/css/mapd_theme.css{,.map}
sass mapd_theme.scss  ../static/css/mapd_theme.css
