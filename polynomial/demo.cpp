#include "polynomial.hpp"

#include <cassert>
#include <cstdio>
#include <unordered_map>

using namespace polynomial;

// Helper: a var-lookup backed by an unordered_map. Unknown ids -> 0.
struct MapLookup {
    const std::unordered_map<int, Fraction>* values;
    Fraction operator()(int id) const {
        auto it = values->find(id);
        return it == values->end() ? Fraction(0) : it->second;
    }
};

static void print(const char* label, Fraction f) {
    std::printf("%-40s = %lld/%lld\n", label,
                (long long)f.num, (long long)f.den);
}

int main() {
    auto ctx = std::make_shared<PolynomialContext>();

    // Register standard +, *, with reciprocal and identities.
    int add_op = ctx->add_binary_operator({
        "+",
        [](Fraction a, Fraction b) { return a + b; },
        [](Fraction a)             { return Fraction(0) - a; },     // unused
        Fraction(0),
    });
    int mul_op = ctx->add_binary_operator({
        "*",
        [](Fraction a, Fraction b) { return a * b; },
        [](Fraction a)             { return Fraction(1) / a; },
        Fraction(1),
    });

    // Variables.
    int x_id = ctx->add_variable("x");
    int y_id = ctx->add_variable("y");

    // Opaque function `f` of arity 2.
    int f_id = ctx->add_opaque_function("f", 2);

    // ---- Test 1: f(x, y) substitution.
    // f(x, y) -> (magic(f,1) + x)^2 + (magic(f,2) + y)^2
    auto x_term = make_var(ctx, add_op, mul_op, x_id);
    auto y_term = make_var(ctx, add_op, mul_op, y_id);
    auto fxy = make_opaque(ctx, add_op, mul_op, f_id, { x_term, y_term });

    // Force the magic-var IDs to exist so we can name them in the env.
    int m1_id = ctx->add_magic_variable(f_id, 1);
    int m2_id = ctx->add_magic_variable(f_id, 2);

    // De-duplication check: re-asking returns the same ids, no new vars.
    int before = ctx->variable_count();
    assert(ctx->add_magic_variable(f_id, 1) == m1_id);
    assert(ctx->add_magic_variable(f_id, 2) == m2_id);
    assert(ctx->variable_count() == before);

    std::unordered_map<int, Fraction> env = {
        {x_id, Fraction(2)},
        {y_id, Fraction(3)},
        {m1_id, Fraction(10)},
        {m2_id, Fraction(20)},
    };
    MapLookup look{&env};

    // Expected: (10+2)^2 + (20+3)^2 = 144 + 529 = 673
    Fraction r1 = fxy->evaluate(look);
    print("f(x=2, y=3) with m1=10 m2=20", r1);
    assert(r1 == Fraction(673));

    // ---- Test 2: Sum(i, 3, f(x, i)).
    int i_id = ctx->add_variable("i");
    auto i_term = make_var(ctx, add_op, mul_op, i_id);
    auto fxi    = make_opaque(ctx, add_op, mul_op, f_id, { x_term, i_term });
    auto bound  = make_var(ctx, add_op, mul_op, /*unused*/ 0); // placeholder
    // bound must be an actual constant -> wrap a product over nothing then tag,
    // but simpler: use a fresh variable bound_v set to 3 in env.
    int b_id = ctx->add_variable("bound_v");
    bound = make_var(ctx, add_op, mul_op, b_id);
    auto sum_expr = make_finite_sum(ctx, add_op, mul_op, i_id, bound, fxi);

    env[i_id] = Fraction(0);     // value gets shadowed inside the sum
    env[b_id] = Fraction(3);

    // i=0: (m1+x)^2 + (m2+0)^2 = (10+5)^2 + 20^2 = 225 + 400 = 625
    // i=1: 225 + (20+1)^2 = 225 + 441 = 666
    // i=2: 225 + (20+2)^2 = 225 + 484 = 709
    // total = 2000
    env[x_id] = Fraction(5);
    Fraction r2 = sum_expr->evaluate(look);
    print("Sum(i,3, f(x=5, i)) with m1=10 m2=20", r2);
    assert(r2 == Fraction(2000));

    // ---- Test 3: Laurent-like negative exponent. x^-2 with x = 4 -> 1/16.
    auto xinv2 = make_var(ctx, add_op, mul_op, x_id, -2);
    env[x_id] = Fraction(4);
    Fraction r3 = xinv2->evaluate(look);
    print("x^-2 with x=4", r3);
    assert(r3 == Fraction(1, 16));

    // ---- Test 4: Tagged passthrough + Product.
    // tag("c", x) * y with x=5, y=3 -> 15
    auto tagged = make_tagged(ctx, add_op, mul_op, "c", x_term);
    auto prod   = make_product(ctx, add_op, mul_op, { tagged, y_term });
    env[x_id] = Fraction(5);
    env[y_id] = Fraction(3);
    Fraction r4 = prod->evaluate(look);
    print("tagged(c, x) * y with x=5 y=3", r4);
    assert(r4 == Fraction(15));

    // ---- Test 5: Empty product/sum -> identities.
    auto empty_prod = make_product(ctx, add_op, mul_op, {});
    auto empty_sum  = make_sum(ctx, add_op, mul_op, {});
    assert(empty_prod->evaluate(look) == Fraction(1));
    assert(empty_sum->evaluate(look) == Fraction(0));
    std::puts("empty product = 1, empty sum = 0  ok");

    std::puts("all tests passed.");
    return 0;
}
