#!/usr/bin/env bash
set -xe
#INSTALL_BASE=opt/omnisci/cpu
INSTALL_BASE=.

cmake --install build --component "exe" --prefix ${PREFIX:-/usr/local}/$INSTALL_BASE
# copy initdb to omnisci_initdb to avoid conflict with psql initdb
mv $PREFIX/$INSTALL_BASE/bin/initdb $PREFIX/$INSTALL_BASE/bin/omnisci_initdb
exit 0

mkdir -p "${PREFIX}/etc/conda/activate.d"
cat > "${PREFIX}/etc/conda/activate.d/${PKG_NAME}_activate.sh" <<EOF
#!/bin/bash
# Avoid cuda and cpu variants of omniscidb in the same environment.
if [[ ! -z "\${PATH_CONDA_OMNISCIDB_BACKUP+x}" ]]
then
  echo "Unset PATH_CONDA_OMNISCIDB_BACKUP(=\${PATH_CONDA_OMNISCIDB_BACKUP}) when activating ${PKG_NAME} from \${CONDA_PREFIX}/${INSTALL_BASE}"
  export PATH="\${PATH_CONDA_OMNISCIDB_BACKUP}"
  unset PATH_CONDA_OMNISCIDB_BACKUP
fi
# Backup environment variables (only if the variables are set)
if [[ ! -z "\${PATH+x}" ]]
then
  export PATH_CONDA_OMNISCIDB_BACKUP="\${PATH:-}"
fi
export PATH="\${PATH}:\${CONDA_PREFIX}/${INSTALL_BASE}/bin"
EOF


mkdir -p "${PREFIX}/etc/conda/deactivate.d"
cat > "${PREFIX}/etc/conda/deactivate.d/${PKG_NAME}_deactivate.sh" <<EOF
#!/bin/bash
# Restore environment variables (if there is anything to restore)
if [[ ! -z "\${PATH_CONDA_OMNISCIDB_BACKUP+x}" ]]
then
  export PATH="\${PATH_CONDA_OMNISCIDB_BACKUP}"
  unset PATH_CONDA_OMNISCIDB_BACKUP
fi
EOF
