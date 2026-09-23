#ifndef HELPERS_IO_HELPERS_H
#define HELPERS_IO_HELPERS_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

namespace helpers {

constexpr std::size_t binary_dump_magic_number() noexcept
{
    return 0xAAAAAAAA55555555;
}

template<class T>
void dump_binary(
    const std::vector<T>& data,
    const std::string& filename) noexcept
{
    std::ofstream ofile(filename, std::ios::binary);

    if(ofile.good())
    {
        const std::size_t magic_number = binary_dump_magic_number();
        const std::size_t t_bytes = sizeof(T);
        const std::size_t size = data.size();

        ofile.write((char *) &magic_number, sizeof(std::size_t));
        ofile.write((char *) &t_bytes, sizeof(std::size_t));
        ofile.write((char *) &size, sizeof(std::size_t));

        ofile.write((char *) data.data(), sizeof(T) * size);
    }
    else
    {
        std::cerr << "Unable to open file." << std::endl;
    }

    ofile.close();
}

template<class T>
std::vector<T> load_binary(
    const std::string& filename,
    std::size_t end = 0,
    std::size_t begin = 0) noexcept
{
    std::vector<T> data;
    std::ifstream ifile(filename, std::ios::binary);

    if(ifile.is_open())
    {
        std::size_t magic_number;

        ifile.read((char *) &magic_number, sizeof(std::size_t));

        if(magic_number == binary_dump_magic_number())
        {
            std::size_t t_bytes;

            ifile.read((char* ) &t_bytes, sizeof(std::size_t));

            if(t_bytes == sizeof(T))
            {
                std::size_t size;

                ifile.read((char* ) &size, sizeof(std::size_t));

                const std::size_t end_ = (end == 0) ? size : end;

                if(begin <= end_ && end_ <= size)
                {
                    ifile.seekg(ifile.tellg() + static_cast<std::streampos>(sizeof(T) * begin));

                    const std::size_t diff = end_ - begin;

                    data.resize(diff);

                    ifile.read((char *) data.data(), sizeof(T) * diff);
                }
                else
                {
                    std::cerr << "Invalid file offsets." << std::endl;
                    data.resize(0);
                }
            }
            else
            {
                std::cerr << "Type mismatch." << std::endl;
                data.resize(0);
            }
        }
        else
        {
            std::cerr << "Invalid file format." << std::endl;
            data.resize(0);
        }
    }
    else
    {
        std::cerr << "Unable to open file." << std::endl;
        data.resize(0);
    }

    ifile.close();

    return data;
}

} // namespace helpers

#endif /* HELPERS_IO_HELPERS_H */
