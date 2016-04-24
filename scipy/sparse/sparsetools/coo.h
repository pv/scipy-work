#ifndef __COO_H__
#define __COO_H__

#include <algorithm>
#include <utility>

extern "C" {
#include <stdint.h>
}


/*
 * C++ iterator and value data structures for handling the COO matrix triples
 */

template <class I, class T> class COOTriple;
template <class I, class T> class COOTripleRef;

// Compare two triples
template <class A, class B>
bool coo_triple_lt(const A& a, const B& b)
{
    if (a.Ai < b.Ai) {
        return true;
    } else if (a.Ai == b.Ai) {
        if (a.Aj < b.Aj) {
            return true;
        }
    }
    return false;
}

/*
 * Mutable reference to a triple (stored in the 3 separate work arrays) 
 */
template <class I, class T>
class COOTripleRef
{
public:
    I &Ai;
    I &Aj;
    T &Ax;

    COOTripleRef(I &i, I &j, T &x)
        : Ai(i), Aj(j), Ax(x)
    {}

    COOTripleRef(const COOTripleRef &other)
        : Ai(other.Ai), Aj(other.Aj), Ax(other.Ax)
    {}

    COOTripleRef& operator=(const COOTripleRef &other) {
        Ai = other.Ai;
        Aj = other.Aj;
        Ax = other.Ax;
        return *this;
    }

    COOTripleRef& operator=(const COOTriple<I,T> &other) {
        Ai = other.Ai;
        Aj = other.Aj;
        Ax = other.Ax;
        return *this;
    }

    bool operator<(const COOTripleRef &other) {
        return coo_triple_lt(*this, other);
    }

    bool operator<(const COOTriple<I,T> &other) {
        return coo_triple_lt(*this, other);
    }
};


/*
 * Copy of a triple (not stored in the work arrays)
 */
template <class I, class T>
class COOTriple
{
public:
    I Ai;
    I Aj;
    T Ax;

    COOTriple(I i, I j, T x)
        : Ai(i), Aj(j), Ax(x)
    {}

    COOTriple(const COOTriple &other)
        : Ai(other.Ai), Aj(other.Aj), Ax(other.Ax)
    {}

    COOTriple(const COOTripleRef<I,T> &other)
        : Ai(other.Ai), Aj(other.Aj), Ax(other.Ax)
    {}

    COOTriple& operator=(const COOTriple &other) {
        Ai = other.Ai;
        Aj = other.Aj;
        Ax = other.Ax;
        return *this;
    }

    COOTriple& operator=(const COOTripleRef<I,T> &other) {
        Ai = other.Ai;
        Aj = other.Aj;
        Ax = other.Ax;
        return *this;
    }

    bool operator<(const COOTriple &other) {
        return coo_triple_lt(*this, other);
    }

    bool operator<(const COOTripleRef<I,T> &other) {
        return coo_triple_lt(*this, other);
    }
};


template <class I, class T>
void swap(COOTripleRef<I,T> a, COOTripleRef<I,T> b)
{
    COOTriple<I,T> tmp(a);
    a = b;
    b = tmp;
}


/*
 * Random access iterator over (i,j,x) tuples in a single COO array.
 * Consists of three related pointers to three arrays.
 */
template <class I, class T>
class COOIterator : public std::iterator<std::random_access_iterator_tag, COOTriple<I,T> >
{
public:
    I *Ai;
    I *Aj;
    T *Ax;

    COOIterator(I *Aip, I *Ajp, T *Axp)
        : Ai(Aip), Aj(Ajp), Ax(Axp)
    {}

    COOTripleRef<I,T> operator*()
    {
        return COOTripleRef<I,T>(*Ai, *Aj, *Ax);
    }

    COOIterator& operator++() {
        ++Ai;
        ++Aj;
        ++Ax;
        return *this;
    }

    COOIterator operator++(int) {
        COOIterator post(*this);
        operator++();
        return post;
    }

    COOIterator& operator--() {
        --Ai;
        --Aj;
        --Ax;
        return *this;
    }

    COOIterator operator--(int) {
        COOIterator post(*this);
        operator--();
        return post;
    }

    COOIterator& operator+=(intptr_t n) {
        Ai += n;
        Aj += n;
        Ax += n;
        return *this;
    }

    COOIterator& operator-=(intptr_t n) {
        Ai -= n;
        Aj -= n;
        Ax -= n;
        return *this;
    }

    COOTripleRef<I,T> operator[](intptr_t n) {
        return *(this + n);
    }
};

template <class I, class T>
COOIterator<I,T> operator+(COOIterator<I,T> a, intptr_t n) {
    return COOIterator<I,T>(a.Ai + n, a.Aj + n, a.Ax + n);
}

template <class I, class T>
COOIterator<I,T> operator-(COOIterator<I,T> a, intptr_t n) {
    return COOIterator<I,T>(a.Ai - n, a.Aj - n, a.Ax - n);
}

template <class I, class T>
COOIterator<I,T> operator+(intptr_t n, COOIterator<I,T> a) {
    return COOIterator<I,T>(n + a.Ai, n + a.Aj, n + a.Ax);
}

template <class I, class T>
intptr_t operator-(COOIterator<I,T> a, COOIterator<I,T> b) {
    return a.Ai - b.Ai;
}

template <class I, class T>
bool operator<(COOIterator<I,T> a, COOIterator<I,T> b) {
    return a.Ai < b.Ai;
}

template <class I, class T>
bool operator>(COOIterator<I,T> a, COOIterator<I,T> b) {
    return a.Ai > b.Ai;
}

template <class I, class T>
bool operator<=(COOIterator<I,T> a, COOIterator<I,T> b) {
    return a.Ai <= b.Ai;
}

template <class I, class T>
bool operator>=(COOIterator<I,T> a, COOIterator<I,T> b) {
    return a.Ai >= b.Ai;
}

template <class I, class T>
bool operator==(COOIterator<I,T> a, COOIterator<I,T> b) {
    return a.Ai == b.Ai;
}

template <class I, class T>
bool operator!=(COOIterator<I,T> a, COOIterator<I,T> b) {
    return a.Ai != b.Ai;
}


/*
 * Compute B = A for COO matrix A, CSR matrix B
 *
 *
 * Input Arguments:
 *   I  n_row      - number of rows in A
 *   I  n_col      - number of columns in A
 *   I  nnz        - number of nonzeros in A
 *   I  Ai[nnz(A)] - row indices
 *   I  Aj[nnz(A)] - column indices
 *   T  Ax[nnz(A)] - nonzeros
 * Output Arguments:
 *   I Bp  - row pointer
 *   I Bj  - column indices
 *   T Bx  - nonzeros
 *
 * Note:
 *   Output arrays Bp, Bj, and Bx must be preallocated
 *
 * Note: 
 *   Input:  row and column indices *are not* assumed to be ordered
 *           
 *   Note: duplicate entries are carried over to the CSR represention
 *
 *   Complexity: Linear.  Specifically O(nnz(A) + max(n_row,n_col))
 * 
 */
template <class I, class T>
void coo_tocsr(const I n_row,
               const I n_col,
               const I nnz,
               const I Ai[],
               const I Aj[],
               const T Ax[],
                     I Bp[],
                     I Bj[],
                     T Bx[])
{
    //compute number of non-zero entries per row of A 
    std::fill(Bp, Bp + n_row, 0);

    for (I n = 0; n < nnz; n++){            
        Bp[Ai[n]]++;
    }

    //cumsum the nnz per row to get Bp[]
    for(I i = 0, cumsum = 0; i < n_row; i++){     
        I temp = Bp[i];
        Bp[i] = cumsum;
        cumsum += temp;
    }
    Bp[n_row] = nnz; 

    //write Aj,Ax into Bj,Bx
    for(I n = 0; n < nnz; n++){
        I row  = Ai[n];
        I dest = Bp[row];

        Bj[dest] = Aj[n];
        Bx[dest] = Ax[n];

        Bp[row]++;
    }

    for(I i = 0, last = 0; i <= n_row; i++){
        I temp = Bp[i];
        Bp[i]  = last;
        last   = temp;
    }

    //now Bp,Bj,Bx form a CSR representation (with possible duplicates)
}

/*
 * Compute B += A for COO matrix A, dense matrix B
 *
 * Input Arguments:
 *   I  n_row           - number of rows in A
 *   I  n_col           - number of columns in A
 *   I  nnz             - number of nonzeros in A
 *   I  Ai[nnz(A)]      - row indices
 *   I  Aj[nnz(A)]      - column indices
 *   T  Ax[nnz(A)]      - nonzeros 
 *   T  Bx[n_row*n_col] - dense matrix
 *
 */
template <class I, class T>
void coo_todense(const I n_row,
                 const I n_col,
                 const I nnz,
                 const I Ai[],
                 const I Aj[],
                 const T Ax[],
                       T Bx[],
		 int fortran)
{
    if (!fortran) {
        for(I n = 0; n < nnz; n++){
            Bx[ (npy_intp)n_col * Ai[n] + Aj[n] ] += Ax[n];
        }
    }
    else {
        for(I n = 0; n < nnz; n++){
            Bx[ (npy_intp)n_row * Aj[n] + Ai[n] ] += Ax[n];
        }
    }
}


/*
 * Compute Y += A*X for COO matrix A and dense vectors X,Y
 *
 *
 * Input Arguments:
 *   I  nnz           - number of nonzeros in A
 *   I  Ai[nnz]       - row indices
 *   I  Aj[nnz]       - column indices
 *   T  Ax[nnz]       - nonzero values
 *   T  Xx[n_col]     - input vector
 *
 * Output Arguments:
 *   T  Yx[n_row]     - output vector
 *
 * Notes:
 *   Output array Yx must be preallocated
 *
 *   Complexity: Linear.  Specifically O(nnz(A))
 * 
 */
template <class I, class T>
void coo_matvec(const I nnz,
	            const I Ai[], 
	            const I Aj[], 
	            const T Ax[],
	            const T Xx[],
	                  T Yx[])
{
    for(I n = 0; n < nnz; n++){
        Yx[Ai[n]] += Ax[n] * Xx[Aj[n]];
    }
}


/*
 * Sum duplicate entries in a COO matrix and sort entries inplace.
 *
 * Input arguments:
 *   I  nnz           - number of nonzeros in A
 *   I  Ai[nnz]       - row indices
 *   I  Aj[nnz]       - column indices
 *   T  Ax[nnz]       - nonzero values
 * Returns:
 *   I  new_nnz       - new nnz
 *
 * Note:
 *   Ai, Aj, Ax will be modified.
 */
template <class I, class T>
I coo_sum_duplicates(const I nnz,
                     I Ai[],
                     I Aj[],
                     T Ax[])
{
    if (nnz <= 1) {
        return nnz;
    }

    /* In-place sort */
    std::sort(COOIterator<I,T>(Ai, Aj, Ax),
              COOIterator<I,T>(Ai + nnz, Aj + nnz, Ax + nnz));

    /* Sum duplicates */
    I k = 0;
    for (I i = 1; i < nnz; ++i) {
        if (Ai[i] == Ai[k] && Aj[i] == Aj[k]) {
            Ax[k] += Ax[i];
        }
        else {
            ++k;
            if (k != i) {
                Ai[k] = Ai[i];
                Aj[k] = Aj[i];
                Ax[k] = Ax[i];
            }
        }
    }
    ++k;

    return k;
}

#endif
