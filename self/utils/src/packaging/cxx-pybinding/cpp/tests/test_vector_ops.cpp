#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fastla/vector_ops.hpp>
#include <vector>

using Catch::Matchers::WithinAbs;

TEST_CASE("dot of small vectors", "[dot]") {
    std::vector<double> x{1, 2, 3}, y{4, 5, 6};
    REQUIRE_THAT(fastla::dot(x.data(), y.data(), 3), WithinAbs(32.0, 1e-12));
}

TEST_CASE("dot of empty vectors is zero", "[dot]") {
    REQUIRE(fastla::dot(nullptr, nullptr, 0) == 0.0);
}

TEST_CASE("norm2 of 3-4-5 triangle", "[norm2]") {
    std::vector<double> x{3, 4};
    REQUIRE_THAT(fastla::norm2(x.data(), 2), WithinAbs(5.0, 1e-12));
}

TEST_CASE("axpy updates y in place", "[axpy]") {
    std::vector<double> x{1, 2, 3}, y{1, 1, 1};
    fastla::axpy(2.0, x.data(), y.data(), 3);
    REQUIRE(y == std::vector<double>{3, 5, 7});
}
