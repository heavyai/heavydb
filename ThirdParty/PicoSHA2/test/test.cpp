#include "../picosha2.h"
#include <iostream>
#include <list>
#include <fstream>

#define PICOSHA2_CHECK_EQUAL(a, b){\
    if(a == b){\
        std::cout << "\033[32m" << __FUNCTION__ << "(LINE:" << __LINE__ << ") is succeeded." << "\033[39m" << std::endl;\
    }\
    else{\
        std::cout << "\033[31m" << __FUNCTION__ << "(LINE:" << __LINE__ << ") is failed.\n\t" << #a << " != " << #b \
        << "\033[39m" << std::endl;\
    }\
}
#define PICOSHA2_CHECK_EQUAL_BYTES(a, b){\
    if(is_same_bytes(a, b)){\
        std::cout << "\033[32m" << __FUNCTION__ << "(LINE:" << __LINE__ << ") is succeeded." << "\033[39m" << std::endl;\
    }\
    else{\
        std::cout << "\033[31m" << __FUNCTION__ << "(LINE:" << __LINE__ << ") is failed.\n\t" << #a << " != " << #b \
        << "\033[39m" << std::endl;\
    }\
}

template<typename InIter1, typename InIter2>
bool is_same_bytes(InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2){
    if(std::distance(first1, last1) != std::distance(first2, last2)){
        return false;
    }
    return std::search(first1, last1, first2, last2) != last1;
}

template<typename InContainer1, typename InContainer2>
bool is_same_bytes(const InContainer1& bytes1, const InContainer2& bytes2){
    return is_same_bytes(bytes1.begin(), bytes1.end(), bytes2.begin(), bytes2.end());
}

template<typename OutIter>
void input_hex(std::istream& is, OutIter first, OutIter last){
    char c;
    std::vector<char> buffer;
    while(first != last){
        if(buffer.size() == 2){
            *(first++) = (buffer.front()*16+buffer.back());
            buffer.clear();
        }
        is >> c;
        if('0' <= c && c <= '9'){
            buffer.push_back(c-'0');
        }else
        if('a' <= c && c <= 'f'){
            buffer.push_back(c-'a'+10); 
        }
    }
}

template<typename OutIter>
void hex_string_to_bytes(const std::string& hex_str, OutIter first, OutIter last){
    assert(hex_str.length() >= 2*std::distance(first, last));
    std::istringstream iss(hex_str);
    input_hex(iss, first, last);
}

template<typename OutContainer>
void hex_string_to_bytes(const std::string& hex_str, OutContainer& bytes){
    hex_string_to_bytes(hex_str, bytes.begin(), bytes.end());
}

typedef std::pair<std::string, std::string> mess_and_hash;
const size_t sample_size = 10;
const std::pair<std::string, std::string> sample_message_list[sample_size] = {
    mess_and_hash("", 
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    mess_and_hash("The quick brown fox jumps over the lazy dog",
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"),
    mess_and_hash("The quick brown fox jumps over the lazy dog.",
            "ef537f25c895bfa782526529a9b63d97aa631564d5d789c2b765448c8635fb6c"),
    mess_and_hash("For this sample, this 63-byte string will be used as input data",
            "f08a78cbbaee082b052ae0708f32fa1e50c5c421aa772ba5dbb406a2ea6be342"),
    mess_and_hash("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
    mess_and_hash("This is exactly 64 bytes long, not counting the terminating byte",
            "ab64eff7e88e2e46165e29f2bce41826bd4c7b3552f6b382a9e7d3af47c245f8"),
    mess_and_hash("This is exactly 64 bytes long, not counting the terminati",
            "46db250ef6d667908de17333c25343778f495d7a8010b9cfa2af97940772e8cd"),
    mess_and_hash("This is exactly 64 bytes long, not counting the terminatin",
            "af38fc14dbbbcc6cd4c9cc73988e1b08373b4e6b04ba61b41f999731185b51af"),
    mess_and_hash("This is exactly 64 bytes long, not counting the terminating",
            "f778b361f650cdd9981ca13adb77f26b8419a407b3938fc54b14e9971045fa9d"),
    mess_and_hash("This is exactly 64 bytes long, not counting the terminating b",
            "9aa72d139c7d7e5a35ea525e2ba6704163555ba647927765a61ccbf12ec60479")
};

void test(){
    for(std::size_t i = 0; i < sample_size; ++i){
        std::string src_str = sample_message_list[i].first;
        std::cout << "src_str: " << src_str  << " size: " << src_str.length() << std::endl;
        std::string ans_hex_str = sample_message_list[i].second;
        std::vector<unsigned char> ans(picosha2::k_digest_size);
        hex_string_to_bytes(ans_hex_str, ans);
        {
            std::vector<unsigned char> hash(picosha2::k_digest_size);
            picosha2::hash256(src_str.begin(), src_str.end(), hash.begin(), hash.end());
            PICOSHA2_CHECK_EQUAL_BYTES(ans, hash);
        }
        {
            std::vector<unsigned char> hash(picosha2::k_digest_size);
            picosha2::hash256(src_str, hash);
            PICOSHA2_CHECK_EQUAL_BYTES(ans, hash);
        }
        {
            std::vector<unsigned char> hash(picosha2::k_digest_size);
            picosha2::hash256(src_str.begin(), src_str.end(), hash);
            PICOSHA2_CHECK_EQUAL_BYTES(ans, hash);
        }
        {
            unsigned char hash_c_array[picosha2::k_digest_size];
            picosha2::hash256(src_str.begin(), src_str.end(), hash_c_array, hash_c_array+picosha2::k_digest_size);
            std::vector<unsigned char> hash(hash_c_array, hash_c_array+picosha2::k_digest_size);
            PICOSHA2_CHECK_EQUAL_BYTES(ans, hash);
        }
        {
            std::list<unsigned char> hash(picosha2::k_digest_size);
            picosha2::hash256(src_str.begin(), src_str.end(), hash.begin(), hash.end());
            PICOSHA2_CHECK_EQUAL_BYTES(ans, hash);
        }
        {
            std::list<unsigned char> hash(picosha2::k_digest_size);
            picosha2::hash256(src_str, hash);
            PICOSHA2_CHECK_EQUAL_BYTES(ans, hash);
        }
        {
            std::string hash_hex_str;
            picosha2::hash256_hex_string(src_str.begin(), src_str.end(), hash_hex_str);
            PICOSHA2_CHECK_EQUAL(ans_hex_str, hash_hex_str);
        }
        {
            std::string hash_hex_str;
            picosha2::hash256_hex_string(src_str, hash_hex_str);
            PICOSHA2_CHECK_EQUAL(ans_hex_str, hash_hex_str);
        }
        {
            std::string hash_hex_str = 
                picosha2::hash256_hex_string(src_str.begin(), src_str.end());
            PICOSHA2_CHECK_EQUAL(ans_hex_str, hash_hex_str);
        }
        {
            std::string hash_hex_str = picosha2::hash256_hex_string(src_str);
            PICOSHA2_CHECK_EQUAL(ans_hex_str, hash_hex_str);
        }
        
        std::vector<unsigned char> src_vect(src_str.begin(), src_str.end());
        {
            std::vector<unsigned char> hash(picosha2::k_digest_size);
            picosha2::hash256(src_vect.begin(), src_vect.end(), hash.begin(), hash.end());
            PICOSHA2_CHECK_EQUAL_BYTES(ans, hash);
        }
        {
            std::vector<unsigned char> hash(picosha2::k_digest_size);
            picosha2::hash256(src_vect, hash);
            PICOSHA2_CHECK_EQUAL_BYTES(ans, hash);
        }
        {
            std::list<unsigned char> hash(picosha2::k_digest_size);
            picosha2::hash256(src_vect.begin(), src_vect.end(), hash.begin(), hash.end());
            PICOSHA2_CHECK_EQUAL_BYTES(ans, hash);
        }
        {
            std::list<unsigned char> hash(picosha2::k_digest_size);
            picosha2::hash256(src_vect.data(), src_vect.data()+src_vect.size(), 
                    hash.begin(), hash.end());
            PICOSHA2_CHECK_EQUAL_BYTES(ans, hash);
        }
        {
            std::list<unsigned char> hash(picosha2::k_digest_size);
            picosha2::hash256(src_vect, hash);
            PICOSHA2_CHECK_EQUAL_BYTES(ans, hash);
        }
        {
            std::string hash_hex_str;
            picosha2::hash256_hex_string(src_vect.begin(), src_vect.end(), hash_hex_str);
            PICOSHA2_CHECK_EQUAL(ans_hex_str, hash_hex_str);
        }
        {
            std::string hash_hex_str;
            picosha2::hash256_hex_string(src_vect, hash_hex_str);
            PICOSHA2_CHECK_EQUAL(ans_hex_str, hash_hex_str);
        }
        {
            std::string hash_hex_str = 
                picosha2::hash256_hex_string(src_vect.begin(), src_vect.end());
            PICOSHA2_CHECK_EQUAL(ans_hex_str, hash_hex_str);
        }
        {
            std::string hash_hex_str = picosha2::hash256_hex_string(src_vect);
            PICOSHA2_CHECK_EQUAL(ans_hex_str, hash_hex_str);
        }
        {
            std::list<char> src(src_str.begin(), src_str.end());
            std::vector<unsigned char> hash(picosha2::k_digest_size);
            picosha2::hash256(src.begin(), src.end(), hash.begin(), hash.end());
            PICOSHA2_CHECK_EQUAL_BYTES(ans, hash);
        }
    }
    {
        picosha2::hash256_one_by_one hasher;
        std::ifstream ifs("test.cpp");
        std::string file_str((std::istreambuf_iterator<char>(ifs)), 
                std::istreambuf_iterator<char>());
        std::size_t i = 0;
        std::size_t block_size = file_str.length()/10;
        for(i = 0; i+block_size <= file_str.length(); i+=block_size){
            hasher.process(file_str.begin()+i, file_str.begin()+i+block_size);
        }
        hasher.process(file_str.begin()+i, file_str.end());
        hasher.finish();
        std::string one_by_one_hex_string;
        get_hash_hex_string(hasher, one_by_one_hex_string);

        std::string hex_string;
        picosha2::hash256_hex_string(file_str.begin(), file_str.end(), hex_string);
        PICOSHA2_CHECK_EQUAL(one_by_one_hex_string, hex_string);

    }
    {
        std::string one_by_one_hex_string; {
            picosha2::hash256_one_by_one hasher;
            std::ifstream ifs("test.cpp");
            std::string file_str((std::istreambuf_iterator<char>(ifs)), 
                    std::istreambuf_iterator<char>());
            std::size_t i = 0;
            std::size_t block_size = file_str.length()/10;
            for(i = 0; i+block_size <= file_str.length(); i+=block_size){
                hasher.process(file_str.begin()+i, file_str.begin()+i+block_size);
            }
            hasher.process(file_str.begin()+i, file_str.end());
            hasher.finish();
            get_hash_hex_string(hasher, one_by_one_hex_string);
        }

        std::ifstream ifs("test.cpp");
        auto first = std::istreambuf_iterator<char>(ifs);
        auto last = std::istreambuf_iterator<char>();
        auto hex_string = picosha2::hash256_hex_string(first, last);
        PICOSHA2_CHECK_EQUAL(one_by_one_hex_string, hex_string);
    }
}

int main(int argc, char* argv[])
{
    test();

    return 0;
}

