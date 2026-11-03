// SPDX-License-Identifier: Apache-2.0
#include "ggml-ssd/store/index.h"
#include "ggml-ssd/platform/file.h"
#include "ggml-ssd/platform/memory.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

static void require(bool ok, const char * message) { if (!ok) { throw std::runtime_error(message); } }

static void write_pack(const std::filesystem::path & path, uint64_t offset, uint64_t data_start = 4096,
                       int32_t expert = 0, uint64_t nbytes = 4096,
                       const std::string & name = "blk.0.ffn_up_exps.weight") {
    std::vector<char> data(8192, 0);
    std::memcpy(data.data(), "GGMLMOEPACKv1", 13);
    const auto put = [&](size_t at, uint64_t value, size_t bytes) {
        for (size_t i = 0; i < bytes; ++i) { data[at + i] = char(value >> (8 * i)); }
    };
    put(16, 1, 4); put(20, 40, 4); put(24, 1, 8); put(32, data_start, 8);
    require(name.size() <= 128, "fixture name too long");
    std::memcpy(data.data() + 40, name.data(), name.size());
    put(168, uint32_t(expert), 4); put(176, offset, 8); put(184, nbytes, 8);
    std::ofstream stream(path, std::ios::binary);
    stream.write(data.data(), data.size());
}

int main() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / ("inferssd-source-index-" + std::to_string(stamp));
    require(std::filesystem::create_directory(directory), "temporary directory failed");
    const auto pack = directory / "expert.pack";
    write_pack(pack, 4096);
    require(bool(ssd::store::source_index::open({pack.string()})), "valid pack rejected");
    write_pack(pack, 0);
    require(!ssd::store::source_index::open({pack.string()}), "pack header published as expert payload");
    write_pack(pack, 4095);
    require(!ssd::store::source_index::open({pack.string()}), "pack padding published as expert payload");
    write_pack(pack, 4096, 40);
    require(!ssd::store::source_index::open({pack.string()}), "pack index overlaps payload boundary");
    write_pack(pack, 4096);
    std::filesystem::resize_file(pack, 200);
    require(!ssd::store::source_index::open({pack.string()}), "missing pack payload published");
    write_pack(pack, 4097);
    require(!ssd::store::source_index::open({pack.string()}), "pack payload extends beyond EOF");
    write_pack(pack, UINT64_MAX);
    require(!ssd::store::source_index::open({pack.string()}), "overflowing pack range published");
    write_pack(pack, 4096, 4096, -1);
    require(!ssd::store::source_index::open({pack.string()}), "negative pack expert published");
    write_pack(pack, 4096, 4096, 0, 0);
    require(!ssd::store::source_index::open({pack.string()}), "empty pack extent published");
    write_pack(pack, 4096, 4096, 0, 4096, "");
    require(!ssd::store::source_index::open({pack.string()}), "empty pack name published");
    write_pack(pack, 4096, 4096, 0, 4096, std::string(128, 'n'));
    require(!ssd::store::source_index::open({pack.string()}), "unterminated pack name published");
    write_pack(pack, 8191, 4096, INT32_MAX, 1, std::string(127, 'n'));
    require(bool(ssd::store::source_index::open({pack.string()})), "valid pack field boundaries rejected");
    std::vector<std::string> aliases;
    for (int seed = 1; seed <= 2; ++seed) {
        const auto data = directory / (std::to_string(seed) + ".bin");
        const auto alias = directory / (std::to_string(seed) + ".tsv");
        std::ofstream stream(data, std::ios::binary);
        for (size_t i = 0; i < 16384; ++i) { stream.put(char((i + seed) % 251)); }
        stream.close();
        std::ofstream(alias) << "tensor\texpert_idx\tnbytes\tsource_path\toffset\n"
            << "blk.0.ffn_up_exps.weight\t3\t5000\t" << data.string() << "\t4097\n";
        aliases.push_back(alias.string());
    }
    auto first = ssd::store::source_index::open({aliases[0]});
    auto second = ssd::store::source_index::open({aliases[1]});
    require(bool(first) && bool(second), "direct source creation failed");
    const auto a = first.value->view(), b = second.value->view();
    int fd_a = -1, fd_b = -1;
    uint64_t offset = 0;
    size_t bytes = 0;
    const char * name = "blk.0.ffn_up_exps.weight";
    require(a.resolve(a.userdata, name, 3, 5000, 4, &fd_a, &offset, &bytes), "first source lookup failed");
    require(b.resolve(b.userdata, name, 3, 5000, 4, &fd_b, &offset, &bytes), "second source lookup failed");
    require(fd_a != fd_b && offset == 4097 && bytes == 5000, "source contexts shared a descriptor or changed logical range");
    require(!b.resolve(b.userdata, name, 3, 5000, 3, &fd_b, &offset, &bytes), "expert-count bound ignored");
    require(!b.resolve(b.userdata, name, 3, 5001, 4, &fd_b, &offset, &bytes), "tensor byte bound ignored");
    first.value.reset();
    uint64_t size = 0;
    require(!ssd::platform::file_size(fd_a, size), "source descriptor survived its owner");
    require(ssd::platform::is_direct(fd_b), "second context lost direct I/O");
    void * buffer = ssd::platform::aligned_alloc_bytes(4096, 8192);
    require(buffer != nullptr, "aligned buffer failed");
    std::unique_ptr<void, decltype(&ssd::platform::aligned_free)> owned(buffer, ssd::platform::aligned_free);
    require(ssd::platform::pread_exact(fd_b, buffer, 8192, 4096), "actual direct read failed");
    const auto * payload = static_cast<unsigned char *>(buffer);
    for (size_t i = 0; i < 5000; ++i) { require(payload[i + 1] == (4097 + i + 2) % 251, "direct payload changed"); }
    second.value.reset();
    const auto bad = directory / "bad.tsv";
    std::ofstream(bad) << "tensor\texpert_idx\tnbytes\tsource_path\toffset\n"
        << name << "\t3\t5000\t" << (directory / "1.bin").string() << "\t16000\n";
    require(!ssd::store::source_index::open({bad.string()}), "out-of-file source published");
    for (const auto & fields : std::vector<std::vector<std::string>>{
            {"abc", "5000", "4097"}, {"4294967296", "5000", "4097"},
            {"18446744073709551616", "5000", "4097"}, {"3", "5000junk", "4097"},
            {"3", "18446744073709551616", "4097"}, {"3", "5000", "junk"},
            {"3", "5000", "-0"}, {"3", "5000", "4097junk"},
            {"3", "5000", "18446744073709551616"}, {"", "5000", "4097"}}) {
        std::ofstream(bad) << "tensor\texpert_idx\tnbytes\tsource_path\toffset\n"
            << name << '\t' << fields[0] << '\t' << fields[1] << '\t'
            << (directory / "1.bin").string() << '\t' << fields[2] << '\n';
        require(!ssd::store::source_index::open({bad.string()}), "malformed numeric alias published");
    }
    require(!ssd::store::source_index::open({aliases[0], (directory / "missing.tsv").string()}), "partial source set published");
    std::ofstream(bad) << "tensor\texpert_idx\tnbytes\tsource_path\toffset\n"
        << name << "\t2147483647\t1\t" << (directory / "1.bin").string() << "\t0\n";
    require(bool(ssd::store::source_index::open({bad.string()})), "valid numeric boundary rejected");
    const auto check_tsv = [&](const std::string & text, bool accepted) {
        std::ofstream stream(bad, std::ios::binary);
        stream.write(text.data(), text.size());
        stream.close();
        require(bool(ssd::store::source_index::open({bad.string()})) == accepted,
                "TSV byte/line validation mismatch");
    };
    const std::string columns = "tensor\texpert_idx\tnbytes\tsource_path\toffset";
    const std::string row = std::string(name) + "\t0\t1\t" + (directory / "1.bin").string() + "\t0";
    check_tsv(columns + "\n" + row + std::string("\0junk", 5) + "\n", false);
    // Put each field last so a C-string truncation would otherwise hide the suffix.
    const std::vector<std::string> names{"tensor", "expert_idx", "nbytes", "source_path", "offset"};
    const std::vector<std::string> values{name, "0", "1", (directory / "1.bin").string(), "0"};
    for (size_t last = 0; last < names.size(); ++last) {
        std::string header, body;
        for (size_t i = 0; i < names.size(); ++i) {
            if (i != last) { header += names[i] + '\t'; body += values[i] + '\t'; }
        }
        header += names[last]; body += values[last];
        check_tsv(header + "\n" + body + std::string("\0junk", 5) + "\n", false);
        check_tsv(header + std::string("\0junk", 5) + "\n" + body + "\n", false);
    }
    check_tsv(columns + "\n" + row, true); // Complete final record need not end in LF.
    check_tsv(columns + "\r\n" + row + "\r\n", true);
    check_tsv(columns + "\n" + row.substr(0, row.size() - 2), false);
    // Unknown columns are allowed, but the entire physical line must fit.
    const std::string padded_row = row + '\t' + std::string(4095 - row.size() - 1, 'x');
    check_tsv(columns + "\textra\n" + padded_row + "\n", true);
    check_tsv(columns + "\textra\n" + padded_row, true);
    check_tsv(columns + "\textra\n" + padded_row + "x\n" + row + "\n", false);
    check_tsv(columns + "\textra\n" + padded_row + "x", false);
    const std::string padded_header = columns + '\t' + std::string(4095 - columns.size() - 1, 'x');
    check_tsv(padded_header + "\n" + row + "\n", true);
    check_tsv(padded_header + "x\n" + row + "\n", false);
#if !defined(_WIN32)
    // A FIFO index or payload must neither block startup nor be published.
    const auto fifo = directory / "fifo";
    require(mkfifo(fifo.c_str(), 0600) == 0, "FIFO creation failed");
    const pid_t child = fork();
    require(child >= 0, "FIFO guard fork failed");
    if (!child) {
        alarm(3);
        if (ssd::store::source_index::open({fifo.string()})) { _exit(1); }
        std::ofstream(bad) << "tensor\texpert_idx\tnbytes\tsource_path\toffset\n"
            << name << "\t3\t5000\t" << fifo.string() << "\t0\n";
        _exit(ssd::store::source_index::open({bad.string()}) ? 2 : 0);
    }
    int child_status = 0;
    require(waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) && !WEXITSTATUS(child_status),
        "non-regular index/payload blocked startup or was published");
#endif
    std::filesystem::remove_all(directory);
    std::puts("SSD_CONTEXT_SOURCE_DIRECT_IO_PASS");
}
