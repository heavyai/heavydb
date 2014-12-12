if [ ! -f ../SqliteConnector/sqlite3.o ]; then
    echo "sqlite3.o not found - building"
    gcc -O3 -c ../SqliteConnector/sqlite3.c
    mv sqlite3.o ../SqliteConnector/
fi

g++ -o catalog_test -g CatalogTest.cpp Catalog.cpp ../SqliteConnector/SqliteConnector.cpp ../SqliteConnector/sqlite3.o -std=c++0x


