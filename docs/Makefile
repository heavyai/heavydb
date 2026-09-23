# SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -j auto
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# breathe-apidoc is optional: generated API pages require a prior Doxygen
# XML tree under ../build/doxygen/xml (for example via the cmake sphinx target).
%: Makefile
	@mkdir -p ${SOURCEDIR}/api
	@find ${SOURCEDIR}/api -mindepth 1 ! -name '.gitkeep' -delete
	@if [ -d ../build/doxygen/xml ]; then \
		breathe-apidoc -o ${SOURCEDIR}/api/ -p HeavyDB ../build/doxygen/xml -g class,struct,union; \
	else \
		echo "WARNING: ../build/doxygen/xml not found; skipping breathe-apidoc."; \
		printf '%s\n' \
			'API Reference' \
			'=============' \
			'' \
			'Generated C++ API pages require Doxygen XML under ``../build/doxygen/xml``.' \
			'Build with the cmake ``sphinx`` target (or generate Doxygen first), then re-run' \
			'``make html`` from this directory.' \
			'' \
			'Until then, see :doc:`../data_model/api` for the Thrift/external API overview.' \
			> ${SOURCEDIR}/api/index.rst; \
	fi
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
