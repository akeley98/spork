#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace polynomial {

// Single typedef hub for the smart-pointer flavor in use.
template<typename T>
using Sptr = std::shared_ptr<T>;

// ---------------------------------------------------------------------------
// 64-bit rational. No overflow handling per the spec.
// ---------------------------------------------------------------------------
struct Fraction {
    int64_t num = 0;
    int64_t den = 1;

    Fraction() = default;
    Fraction(int64_t n) : num(n), den(1) {}
    Fraction(int64_t n, int64_t d) : num(n), den(d) { normalize(); }

    static int64_t gcd64(int64_t a, int64_t b) {
        if (a < 0) a = -a;
        if (b < 0) b = -b;
        while (b != 0) { int64_t t = a % b; a = b; b = t; }
        return a;
    }

    void normalize() {
        if (den < 0) { num = -num; den = -den; }
        int64_t g = gcd64(num, den);
        if (g > 0) { num /= g; den /= g; }
    }
};

inline Fraction operator+(Fraction a, Fraction b) {
    return Fraction(a.num * b.den + b.num * a.den, a.den * b.den);
}
inline Fraction operator-(Fraction a, Fraction b) {
    return Fraction(a.num * b.den - b.num * a.den, a.den * b.den);
}
inline Fraction operator*(Fraction a, Fraction b) {
    return Fraction(a.num * b.num, a.den * b.den);
}
inline Fraction operator/(Fraction a, Fraction b) {
    return Fraction(a.num * b.den, a.den * b.num);
}
inline bool operator==(Fraction a, Fraction b) {
    return a.num * b.den == b.num * a.den;
}
inline bool operator!=(Fraction a, Fraction b) { return !(a == b); }

// ---------------------------------------------------------------------------
// Polynomial context. Holds the registered names/IDs for variables,
// opaque functions, and binary operators.
// ---------------------------------------------------------------------------

// A binary operator bundles all callables needed to treat it like a ring
// operation. Extend this struct as new operations become necessary.
struct BinaryOperator {
    std::string name;
    std::function<Fraction(Fraction, Fraction)> apply;
    std::function<Fraction(Fraction)> reciprocal;  // for negative exponents
    Fraction identity;                              // for empty fold / zero exponent
};

struct OpaqueFunction {
    std::string name;
    int arity;
};

class PolynomialContext {
public:
    int add_variable(std::string name) {
        variables_.push_back(std::move(name));
        return (int)variables_.size() - 1;
    }

    int add_opaque_function(std::string name, int arity) {
        opaque_funcs_.push_back({std::move(name), arity});
        return (int)opaque_funcs_.size() - 1;
    }

    int add_binary_operator(BinaryOperator op) {
        binops_.push_back(std::move(op));
        return (int)binops_.size() - 1;
    }

    // De-duplicates on (opaque_func_id, i). Same pair always returns the
    // same variable id.
    int add_magic_variable(int opaque_func_id, int i) {
        auto key = std::make_pair(opaque_func_id, i);
        auto it = magic_map_.find(key);
        if (it != magic_map_.end()) return it->second;
        std::string name = "magic(" + opaque_funcs_[opaque_func_id].name
                         + ", " + std::to_string(i) + ")";
        int id = add_variable(std::move(name));
        magic_map_.emplace(key, id);
        return id;
    }

    const std::string&    variable_name(int id)    const { return variables_[id]; }
    const OpaqueFunction& opaque_function(int id)  const { return opaque_funcs_[id]; }
    const BinaryOperator& binary_operator(int id)  const { return binops_[id]; }

    int variable_count()         const { return (int)variables_.size(); }
    int opaque_function_count()  const { return (int)opaque_funcs_.size(); }
    int binary_operator_count()  const { return (int)binops_.size(); }

private:
    std::vector<std::string>      variables_;
    std::vector<OpaqueFunction>   opaque_funcs_;
    std::vector<BinaryOperator>   binops_;
    std::map<std::pair<int,int>, int> magic_map_;
};

using PolynomialContextPtr = Sptr<PolynomialContext>;

// ---------------------------------------------------------------------------
// Extended polynomial.
// ---------------------------------------------------------------------------

struct ExtendedPolynomial;
using ExtPolyPtr = Sptr<ExtendedPolynomial>;

struct VarTerm {
    int var_id;
    int exponent;
};
struct OpaqueTerm {
    int opaque_func_id;
    std::vector<ExtPolyPtr> args;
    int exponent;
};
struct ProductTerm {
    std::vector<ExtPolyPtr> factors;
};
struct SumTerm {
    std::vector<ExtPolyPtr> terms;
};
struct FiniteSumTerm {
    int var_id;
    ExtPolyPtr bound;
    ExtPolyPtr expr;
};
struct TaggedTerm {
    std::string tag;
    ExtPolyPtr value;
};

struct ExtendedPolynomial {
    PolynomialContextPtr context;
    int ring_sum_op;
    int ring_product_op;
    std::variant<VarTerm, OpaqueTerm, ProductTerm, SumTerm, FiniteSumTerm, TaggedTerm> state;

    template<typename VarLookup>
    Fraction evaluate(const VarLookup& var_lookup) const;
};

// Constructor helpers ------------------------------------------------------

inline ExtPolyPtr make_var(PolynomialContextPtr ctx, int sum_op, int prod_op,
                           int var_id, int exp = 1) {
    auto p = std::make_shared<ExtendedPolynomial>();
    p->context = std::move(ctx);
    p->ring_sum_op = sum_op;
    p->ring_product_op = prod_op;
    p->state = VarTerm{var_id, exp};
    return p;
}

inline ExtPolyPtr make_opaque(PolynomialContextPtr ctx, int sum_op, int prod_op,
                              int func_id, std::vector<ExtPolyPtr> args, int exp = 1) {
    auto p = std::make_shared<ExtendedPolynomial>();
    p->context = std::move(ctx);
    p->ring_sum_op = sum_op;
    p->ring_product_op = prod_op;
    p->state = OpaqueTerm{func_id, std::move(args), exp};
    return p;
}

inline ExtPolyPtr make_product(PolynomialContextPtr ctx, int sum_op, int prod_op,
                               std::vector<ExtPolyPtr> factors) {
    auto p = std::make_shared<ExtendedPolynomial>();
    p->context = std::move(ctx);
    p->ring_sum_op = sum_op;
    p->ring_product_op = prod_op;
    p->state = ProductTerm{std::move(factors)};
    return p;
}

inline ExtPolyPtr make_sum(PolynomialContextPtr ctx, int sum_op, int prod_op,
                           std::vector<ExtPolyPtr> terms) {
    auto p = std::make_shared<ExtendedPolynomial>();
    p->context = std::move(ctx);
    p->ring_sum_op = sum_op;
    p->ring_product_op = prod_op;
    p->state = SumTerm{std::move(terms)};
    return p;
}

inline ExtPolyPtr make_finite_sum(PolynomialContextPtr ctx, int sum_op, int prod_op,
                                  int var_id, ExtPolyPtr bound, ExtPolyPtr expr) {
    auto p = std::make_shared<ExtendedPolynomial>();
    p->context = std::move(ctx);
    p->ring_sum_op = sum_op;
    p->ring_product_op = prod_op;
    p->state = FiniteSumTerm{var_id, std::move(bound), std::move(expr)};
    return p;
}

inline ExtPolyPtr make_tagged(PolynomialContextPtr ctx, int sum_op, int prod_op,
                              std::string tag, ExtPolyPtr value) {
    auto p = std::make_shared<ExtendedPolynomial>();
    p->context = std::move(ctx);
    p->ring_sum_op = sum_op;
    p->ring_product_op = prod_op;
    p->state = TaggedTerm{std::move(tag), std::move(value)};
    return p;
}

// Evaluation ---------------------------------------------------------------

// Raise `value` to integer `exp` using a BinaryOperator's apply/reciprocal/identity.
inline Fraction apply_exponent(const BinaryOperator& op, Fraction value, int exp) {
    if (exp == 0) return op.identity;
    bool neg = exp < 0;
    if (neg) exp = -exp;
    Fraction result = value;
    for (int i = 1; i < exp; ++i) result = op.apply(result, value);
    if (neg) result = op.reciprocal(result);
    return result;
}

template<typename VarLookup>
Fraction ExtendedPolynomial::evaluate(const VarLookup& var_lookup) const {
    const auto& sum_op  = context->binary_operator(ring_sum_op);
    const auto& prod_op = context->binary_operator(ring_product_op);

    return std::visit([&](const auto& s) -> Fraction {
        using T = std::decay_t<decltype(s)>;

        if constexpr (std::is_same_v<T, VarTerm>) {
            return apply_exponent(prod_op, var_lookup(s.var_id), s.exponent);

        } else if constexpr (std::is_same_v<T, OpaqueTerm>) {
            // Substitution: A(x1, ..., xn) ->
            //   (magic(A,1) + x1)^2 + (magic(A,2) + x2)^2 + ...
            // with + = ring sum, ^2 = ring product applied to itself.
            Fraction acc = sum_op.identity;
            for (size_t i = 0; i < s.args.size(); ++i) {
                int magic_id = context->add_magic_variable(s.opaque_func_id, (int)(i + 1));
                Fraction m  = var_lookup(magic_id);
                Fraction x  = s.args[i]->evaluate(var_lookup);
                Fraction inner = sum_op.apply(m, x);
                Fraction sq    = prod_op.apply(inner, inner);
                acc = sum_op.apply(acc, sq);
            }
            return apply_exponent(prod_op, acc, s.exponent);

        } else if constexpr (std::is_same_v<T, ProductTerm>) {
            Fraction acc = prod_op.identity;
            for (const auto& f : s.factors) {
                acc = prod_op.apply(acc, f->evaluate(var_lookup));
            }
            return acc;

        } else if constexpr (std::is_same_v<T, SumTerm>) {
            Fraction acc = sum_op.identity;
            for (const auto& t : s.terms) {
                acc = sum_op.apply(acc, t->evaluate(var_lookup));
            }
            return acc;

        } else if constexpr (std::is_same_v<T, FiniteSumTerm>) {
            Fraction bound_val = s.bound->evaluate(var_lookup);
            if (bound_val.den != 1) {
                throw std::runtime_error("FiniteSum bound is not an integer");
            }
            int64_t bound_n = bound_val.num;
            int sum_var_id = s.var_id;
            Fraction acc = sum_op.identity;
            // The inner lookup is type-erased to std::function so that the
            // recursive evaluate<> instantiation does not produce a fresh
            // template type per nesting level (which would blow up the
            // compiler).
            int64_t k = 0;
            std::function<Fraction(int)> inner_lookup =
                [&](int id) -> Fraction {
                    return id == sum_var_id ? Fraction(k) : var_lookup(id);
                };
            for (; k < bound_n; ++k) {
                acc = sum_op.apply(acc, s.expr->evaluate(inner_lookup));
            }
            return acc;

        } else if constexpr (std::is_same_v<T, TaggedTerm>) {
            return s.value->evaluate(var_lookup);
        }
    }, state);
}

} // namespace polynomial
