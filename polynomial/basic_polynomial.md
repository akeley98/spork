I need to create a quick-and-dirty C++ prototype for some experiments on using polynomials for symbolic evaluation.
I would prefer to avoid using libraries like FLINT due to reasons that won't be evident at first.

The starting point is you need to implement 64-bit fractions.
Don't worry about what will happen if there is overflow.

Then I need you to create a "polynomial context" that contains
    * A resizable array of variables (with string names)
    * A resizable array of opaque functions (they have a string name and an integer arity)
    * A resizable array of binary operators (it's a string name and a binary function `fraction × fraction -> fraction`)
The public interface allows adding to those arrays, and looking them up by integer ID.
It must also allow adding magical variables auto-named based on an (opaque function, int) pair.
This last interface must de-duplicate based on the value of that pair!

Everything is equivalent by ID equality; the strings are just for show and never actually compared.

From there, please implement what I call "extended polynomials".
This contains a reference-counted `shared_ptr` to the polynomial context (please use a typedef for `shared_ptr`, as I may want to replace it later on).
It also contains a "ring sum operator" and "ring product operator", each a binary operator from the context.
Finally, it contains one of the following as the mathematical state:
    * A variable (int ID of variable), with integer exponent (positive or negative, like Laurent polynomials)
    * Evaluation of an opaque function (int ID of opaque function, array of extended polynomials as parameters), also with integer exponent
    * A product of a fixed number of extended polynomials (evaluated using "ring product operator")
    * A sum of a fixed number of extended polynomials (evaluated using "ring sum operator")
    * A finite summation: `Sum(variable_name, bound, expr)` where `bound` and `expr` are both extended polynomials. Sum from `variable_name = 0` to `variable_name = bound - 1` of `expr` evaluated with `variable_name` substituted. Also using "ring sum operator".
    * Tagged value. (string tag, extended polynomial) pair. Evaluated as just the value of the extended polynomial inside.

Implement an evaluation function for the extended polynomial with fraction arguments.
For opaque functions A(x1, x2, ...) please substitute with

        (magic(A, 1) + x1) ^ 2 + (magic(A, 2) + x2) ^ 2 + ...

where `magic(A, i)` is the magical variable mentioned earlier.
