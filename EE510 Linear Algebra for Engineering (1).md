# EE510 Linear Algebra for Engineering

![USC Viterbi School of Engineering](https://viterbischool.usc.edu/wp-content/uploads/2021/04/USC-Viterbi-School-of-Engineering.png)

## Week 1 Session 1

Review: 

### Logical Inference

Logical Statement P and Q

| $$P$$ | $$Q$$ | $$\neg P$$ | $$\neg Q$$ | $$P \and Q$$ | $$P \or Q$$ | $$P \implies Q$$ | $$P \iff Q$$ |
| ----- | ----- | ---------- | ---------- | ------------ | ----------- | ---------------- | ------------ |
| 1     | 1     | 0          | 0          | 1            | 1           | 1                | 1            |
| 0     | 1     | 1          | 0          | 0            | 1           | 1                | 0            |
| 1     | 0     | 0          | 1          | 0            | 1           | 0                | 0            |
| 0     | 0     | 1          | 1          | 0            | 0           | 1                | 1            |

$$\and$$ is AND

$$\or$$ is OR

$$\implies$$ If then

$$\iff$$ If and only if

Conditional: $$P \implies Q$$

Contrastive: $$\neg P \implies \neg Q$$

Converse: $$ Q \implies P$$

Predicate: $$Px$$ means $$ x $$ is  $$P$$

Quantifier: $$ \forall x$$ (universal) means "for all $$x$$"

​		    $$\exists x$$ (existential) means "for some $$x$$"

$$\forall x$$: $$Px$$ means "Everything is $$P$$"

$$Px_1 \:AND\: Px_2 \:AND\: Px_3 \:AND$$

$$\exists x$$: $$Px$$ means "Something is $$P$$"

$$Px_1 \:OR\: Px_2 \:OR\: Px_3 \:OR$$

<u>Rules of Inference:</u>

- Modus Ponens: Affirming the antecedent

Premise 1: $$P \implies Q$$

Premise 2: $$P$$

Conclusion: $$Q$$

- Modus Tollens: Denying the consequent

Premise 1: $$P \implies Q$$

Premise 2: $$ \neg Q$$

Conclusion: $$\neg Q$$

- Mathematical Induction

Goal: Proof that $$P_n \forall n\geq n_0$$ where $$n_0$$ is usually 0 or a positive number

1. Basis step: $$P_{n0}$$

2. Induction step:

   $$P_{n0} \: \& \: P_{n-1} \implies P_n$$

   Assume $$P_{n0}$$ and $$P_{n-1}$$ then show $$P_n$$

### Set Theory

set: a collection of elements

$$ x \in A$$, where $$x$$ is element, $$A$$ is set, $$\in \equiv$$ Element hood  

$$A=\{a_1,a_2,...,a_n\}$$

Subset: $$A \subset X$$, $$B \subset X$$

$$A \subset X$$ if and only if $$\forall x \in A$$, $$x \in A$$

$$A^c=\{x \in X: X \notin A\}$$

$$ A \bigcup B = \{x \in X: x \in A \: OR \: x \in B\}$$

$$ A \bigcap B = \{x \in X: x \in A \: AND \: x \in B\}$$

De Morgan's Law: 

$$A \bigcup B =(A^c \bigcap B^c)^c$$

$$A \bigcap B =(A^c \bigcup B^c)^c$$

### Function

$$f: X \implies Y$$

![image-20240914121408972](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240914121408972.png)

Injective function:

![image-20240914121427647](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240914121427647.png)

$$f$$ is injective if and only if $$\forall x, z \in X$$,  $$f(x)=f(z) \: \implies x=z$$

Surjective function:

![image-20240914121523272](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240914121523272.png)

$$\forall y \in Y, \exists x \in X: f(x)=y$$

Bijective Function (1-1 correspondence)

![image-20240914121648663](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240914121648663.png)

$$f$$ is bijective if and only if $$f$$ is injective and surjective.

### Cardinality of a set

Finite set:

$$A=\{a_1,..., a_n\}$$, where $$n\in \Z^{+}$$

Infinite set:

1. Uncountably infinite

   $$\R$$

2. Countably infinite

   $$\Z^{+}$$

Example: 

$$f: \Z^{+} (1-1 \: correspondence) \implies \Z^{-}$$

### Vectors

A vector is a 1-dimensional array of scalars over a field. 

Let $$V= \in \R^(n): v_1, ..., v_n \in \R$$

For $$u, v \in \R^{n}$$

- Vector Addition: 

$$u+v=\begin{bmatrix}
u_1+v_1\\...\\u_n+v_n
\end{bmatrix} \in \R^{n}$$

- Scalar Multiplication:

For $$a \in \R, v \in R^{n}$$

Then $$av=\begin{bmatrix}
av_1\\...\\av_n
\end{bmatrix}$$

- Linear Combination:

For $$a,b \in \R$$ and $$u,v \in \R^{n}$$

$$au+bv=\begin{bmatrix}
au_1\\...\\au_n
\end{bmatrix}+ \begin{bmatrix}
bv_1\\...\\bv_n
\end{bmatrix}=\begin{bmatrix}
au_1+bv_1\\...\\au_n+bv_n
\end{bmatrix}$$

![image-20240914123538823](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240914123538823.png)

- Inner Product

$$ u, v \in \R^{n}$$ 

$$u \cdot v=\sum_{k=1}^{n}u_kv_k$$

Length:

$$||u||^{2}=u \cdot u = \sum_{k=1}^{n}(u_k)^2$$

$$||u_k||^2 = \sqrt(\sum_{k=1}^{n}(u_k)^2)$$

![image-20240914123829751](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240914123829751.png)

$$u \cdot v =||u|| \: ||v|| cos(\theta)$$

$$||v||^{2}=v \cdot v $$

$$cos(\theta)=\frac{u \cdot v}{||u|| \: ||v||}$$

------

Thm: Cauchy Schwartz Inequality

Let $$u,v \in \R^{n}$$, $$|u\cdot v| \leq ||u|| \: ||v||$$

Proof:

Case 1: $$||u||=0 $$ or $$||v||=0 $$

If $$||u||=0 $$ : $$|0 \cdot v|=0 \leq ||u|| \: ||v||=0||v||=0$$ 

If $$||v||=0 $$ : $$|u \cdot 0|=0 \leq ||u|| \: ||v||=||u||0=0$$ 



Case 2: $$||u|| \neq 0$$ and $$||v|| \neq 0$$



Lemma 1: If $$ a,b \in \R$$, then $$a^2+b^2 \geq 2ab$$

Proof: $$(a-b)^2 \geq 0$$ for $$a,b \in \R$$

$$a^2+b^2-2ab \geq 0$$

$$a^2+b^2 \geq 2ab$$



Lemma 2: If If $$ a,b \in \R$$, then $$a^2+b^2 \geq -2ab$$

Proof: $$(a+b)^2 \geq 0$$ for $$a,b \in \R$$

$$a^2+b^2+2ab \geq 0$$

$$a^2+b^2 \geq -2ab$$



Let $$a_k \equiv \frac{u_k}{||u||}$$, $$b_k \equiv \frac{v_k}{||v||}$$

$$(a_k)^2+(b_k)^2 \geq 2a_kb_k$$            using  Lemma 1

$$\sum_{k=1}^{n}(\frac{(u_k)^2}{(||u||)^2}+\frac{(v_k)^2}{(||v||)^2}) \geq \sum_{k=1}^{n}(2\frac{u_k}{||u||}\frac{v_k}{||v||})$$

$$\frac{1}{(||u||)^2}  \sum_{k=1}^{n}{(u_k)^2}+\frac{1}{(||v||)^2}  \sum_{k=1}^{n}{(v_k)^2} \geq \frac{2}{||u||||v||} \sum_{k=1}^{n}u_kv_k$$

$$\frac{(||u||)^2}{(||u||)^2}+\frac{(||v||)^2}{(||v||)^2} \geq \frac{2}{||u||\:||v||}(u \cdot v)$$

$$2 \geq \frac{2}{||u||\:||v||}(u \cdot v)$$

$$||u||\:||v|| \geq (u \cdot v)$$



Similarly,

$$||u||\:||v|| \geq -(u \cdot v)$$            using  Lemma 2   

Therefore $$||u||\:||v||=0$$

## Week 1 Session 2

### Outline

Vectors: Dot Products, Norm, Minkowski Inequality

Matrices: Matrix multiplication ,Transpose, Trace, Block matrices

------

$$u,v \in  \R^{n}$$

Inner Product: $$u \cdot v= \sum_{k=1}^{n} u_kv_k$$

Length (Norm): $$||u||^2=u \cdot c=\sum_{k=1}^{n}(u_k)^2$$

Properties: For $$k \in \R$$, $$u,v,w \in \R^{n}$$

1. $$u \cdot v= v \cdot u$$
2. $$u \cdot (v+w)=(u \cdot v)+(u \cdot w)$$
3. $$ku \cdot v=k(u \cdot v)$$
4. $$u \cdot u \geq 0$$ and $$u \cdot u =0$$ if and only if $$u=\mathbf{0}$$

$$|u \cdot v| \leq ||u|| \: ||v||$$ : Cauchy Schwartz Inequality

Minkowski Inequality

$$||u+v|| \leq ||u||+||v||$$

Proof: 

$$||u+v||^2=(u+v) \cdot (u+v)$$

$$=(u \cdot u)+(u \cdot v)+(v \cdot u)+(v \cdot v)$$

$$=||u||^2+2(u \cdot v)+||v||^2$$

$$\leq ||u||^2+2|u \cdot v|+||v||^2$$           $$(u \cdot v) \in \R$$

$$\leq ||u||^2+2||u||\:||v||+||v||^2$$       Cauchy Schwartz Inequality

$$=(||u||+||v||)^2$$

Therefore:

Triangle Inequality 

$$||u+v||^2 \leq (||u||+||v||)^2$$

$$||u+v|| \leq ||u||+||v||$$

------

$$u$$ nd $$v$$ are orthogonal (perpendicular) $$\implies u \cdot v=0$$

Normalizing a vector: $$\frac{v}{||v||}$$

![image-20240914131301709](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240914131301709.png)

$$u^{\ast} \equiv$$ Projection of $$u$$ onto $$v$$

$$u^{\ast} \equiv Proj(u,v)=\frac{u \cdot v}{||v||^2}v$$

$$u^{\ast} \equiv Proj(u,v)=||u|| \frac{v}{||v||}$$, where $$||u||$$ is the magnitude, $$\frac{v}{||v||}$$ is the direction

$$=||u||cos(\theta)\frac{v}{||v||}$$

$$=||u|| \: ||v|| cos(\theta) \frac{v}{||v||^2}$$

$$=\frac{u \cdot v}{||v||^2}v$$

------

### Complex Vectors

$$u,v \in \C^{n}$$

$$u \cdot v= \sum_{k=1}^{n} u_k v_k^\star$$

where $$v_k \in \C$$, $$v_k=a_k+jb_k$$, where $$a_k$$ is the real part, and $$b_k$$ is the imaginary part

### Matrices

$$A \equiv [a_ij] = \begin{bmatrix}
a_{11} & a_{12} & ... & a_{1n} \\ ... &...&...&...\\ a_{m1} & a_{m2} & ... & a_{mn}\\\end{bmatrix}$$

$$A$$ is $$m \times n$$ with $$m$$ rows and $$n$$ columns

$$A \in \R^{m \times n}$$

$$A=\begin{bmatrix} 2 & 1 & 0 \\ 4 & 2 &-1\\ 3 & 3 & 0\\2 & 4 & 2\\\end{bmatrix} \in \R^{4 \times 3}$$

A row vector: $$v=[v_1, v_2, ..., v_n] \in K^{1 \times n}$$

A column vector: $$v = \begin{bmatrix} v_1 \\  v_2\\...\\v_m\end{bmatrix} \in K^{m \times 1}$$

------

Matrix Addition

$$A, B \in K^{m \times n}$$$$A+B=\begin{bmatrix}
a_{11} & a_{12} & ... & a_{1n} \\ ... &...&...&...\\ a_{m1} & a_{m2} & ... & a_{mn}\\\end{bmatrix} + \begin{bmatrix}
b_{11} & b_{12} & ... & b_{1n} \\ ... &...&...&...\\ b_{m1} & b_{m2} & ... & b_{mn}\\\end{bmatrix}$$

$$=\begin{bmatrix}
a_{11}+b_{11} & a_{12}+b_{12} & ... & a_{1n}+b_{1n} \\ ... &...&...&...\\ a_{m1}+b_{m1} & a_{m2}+b_{m2} & ... & a_{mn}+b_{mn}\\\end{bmatrix} $$

Scalar Multiplication

If $$k \in K, A \in K^{m \times n}$$

$$kA=\begin{bmatrix}
ka_{11} & ka_{12} & ... & ka_{1n} \\ ... &...&...&...\\ ka_{m1} & ka_{m2} & ... & ka_{mn}\\\end{bmatrix}$$

Null Matrix

$$A \equiv [a_{ij}]=\mathbf{0}$$

$$\forall i, j, a_{ij}=0$$

$$A= \begin{bmatrix}
0 & 0 & ... & 0 \\ ... &...&...&...\\ 0 & 0 & ... & 0\\\end{bmatrix}$$

Linear Combination:

$$a,b \in K$$ , $$A, B \in K^{M \times N}$$

$$aA+bB=\begin{bmatrix}aa_{11}+bb_{11} & aa_{12}+bb_{12} & ... & aa_{1n}+bb_{1n} \\ ... &...&...&...\\ aa_{m1}+bb_{m1} & aa_{m2}+bb_{m2} & ... & aa_{mn}+bb_{mn}\\\end{bmatrix} $$

Properties: 

If $$k, k' \in K$$ and $$A,B,C \in K^{m \times n}$$

1. $$A+B=B+A$$                             Commutativity
2. $$A+(B+C)=(A+B)+C$$     Associativity
3. $$k(A+B)=kA+kB$$
4. $$kk'A=k(k'A)$$
5. $$A + -A= 0$$
6. $$A+0=A$$

Transpose:

If $$A \in K^{m \times n}$$ and $$A=[a_{ij}]$$, then

$$A^T \in K^{n \times m}$$ and $$A^T=B=[b_{ij}]$$ where $$b_{ij}=a_{ji}$$

$$A=\begin{bmatrix}
a_{11} & a_{12} & ... & a_{1n} \\ ... &...&...&...\\ a_{m1} & a_{m2} & ... & a_{mn}\\\end{bmatrix}$$ where dimension is $$m \times n$$

 $$A^T=\begin{bmatrix}
a_{11} & a_{12} & ... & a_{m1} \\ ... &...&...&...\\ a_{1n} & a_{2n} & ... & a_{mn}\\\end{bmatrix}$$ where dimension is $$n \times m$$

Example. 

$$A=\begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \\\end{bmatrix}$$

then

$$A^T=\begin{bmatrix} 1 & 3 & 5\\ 2 & 4 & 6 \\\end{bmatrix}$$

Properties:

If $$A,B \in K^{m \times n}$$

1. $$(A+B)^T=A^T+B^T$$
2. $$(A^T)^T=A$$

Let $$u,v \in K^{m \times 1}$$

then $$u \cdot v= u^Tv$$

Square Matrix

$$A=[a_{ij}]$$ is a square matrix if and only if the number of rows equal the number of columns.

$$m=n$$

$$A=\begin{bmatrix}
a_{11} & a_{12} & ... & a_{1n} \\ ... &...&...&...\\ a_{n1} & a_{n2} & ... & a_{nn}\\\end{bmatrix}$$

Diagonal Matrix:

$$A \equiv [a_{ij}]$$

A square matrix such that $$\forall i \neq j$$, $$a_{ij}=0$$

$$A=\begin{bmatrix}
a_{11} & 0 & ... & 0 \\ ... &...&...&...\\ 0 & 0 & ... & a_{nn}\\\end{bmatrix}$$

Triangular Matrices: 

Upper triangular

$$A=\begin{bmatrix}
a_{11} & a_{12} & ... & a_{1n} \\ ... &...&...&...\\ 0 & 0 & ... & a_{nn}\\\end{bmatrix}$$

$$\forall i>j, a_{ij}=0$$

Lower triangular

$$A=\begin{bmatrix}
a_{11} & 0 & ... & 0 \\ ... &...&...&...\\ a_{n1} & a_{n2} & ... & a_{nn}\\\end{bmatrix}$$

$$\forall i<j, a_{ij}=0$$

------

Matrix Multiplication

| $$A$$          | $$B$$           | $$C$$          |
| -------------- | --------------- | -------------- |
| $$m \times n$$ | $$ n \times p$$ | $$m \times p$$ |
| $$[a_{ij}]$$   | $$[b_{ij}]$$    | $$[c_{ij}]$$   |

$$c_{ij}=\sum_{k=1}^{n}a_{ik}b_{kj}$$

| $$B$$          | $$A$$          | $$D$$          |
| -------------- | -------------- | -------------- |
| $$n \times m$$ | $$m \times p$$ | $$n \times p$$ |
| $$[a_{ij}]$$   | $$[b_{ij}]$$   | $$[c_{ij}]$$   |

$$BA=D$$

$$d_{ij}=\sum_{k=1}^{m}b_{ik}a_{kj}$$

$$c_{ij}= i^{th} \: row\: of \: A \:  \cdot j^{th} \: column \: of \: B$$

$$i^{th} \: row\: of \: A$$: $$A_{i:}$$

$$j^{th} \: column \: of \: B$$: $$B_{:j}$$

![image-20240914182321007](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240914182321007.png)

Properties: If $$A,B,C$$ are conformable for multiplication

1. $$(AB)C=A(BC)$$                   Associativity
2. $$A(B+C)=AB+AC$$        Left distribution
3. $$(A+B)C=AC+BC$$        Right distribution
4. $$(AB)^T=B^TA^T$$                     
5. $$c(AB)=(cA)B=A(cB)$$  if $$c$$ is a scalar
6. $$ AB \neq BA$$

------

### Trace

$$A \in K^{n \times n}$$

$$Tr(A)= \sum_{k=1}^{n}a_{kk}$$

$$A=\begin{bmatrix} 1 & 2 & 3 \\ 0 & 1 &4\\ 0 & 2 & 5\\\end{bmatrix}$$

$$Tr(A)=a_{11}+a_{22}+a_{33}=1+1+5=7$$

Properties:

If $$A,B,C$$ are conformable for multiplication

1. $$Tr(A)=Tr(A^T)$$
2. $$Tr(BA)=Tr(AB)$$
3. $$Tr(ABC)=Tr(BCA)=Tr(CAB)$$

------

Cyclic Property of Trace

Thm:

$$Tr(A_1 \: A_2\: ... \: A_{n-1}\: A_n)=Tr(A_n \: A_1\: ... \: A_{n-1})$$

If the matrices $$A_k$$ are conformable for matrix multiplication where $$T_r$$ is the trace operator:

$$Tr(A)=\sum_{k=1}^{p}a_{kk}$$ if $$A$$ is a square matrix

$$A_k \in \C^{m_k \times n_k}$$

Proof:

Lemma 1:

$$Tr(AB)=Tr(BA)$$

Lemma 2:

$$A \times (B \times C)=(A \times B) \times C$$

------

Lemma 1:

$$A$$ dimension is $$ m \times n$$ 

$$B$$ dimension is $$n \times m$$

Then $$A \times B$$ is $$m \times m$$, $$B \times A$$ is $$ n \times n$$

$$Tr(AB)=\sum_{k=1}^{m}(AB)_{kk}$$       def of $$Tr$$

$$=\sum_{k=1}^{m}(\sum_{l=1}^{n}a_{kl}b_{lk})$$             def of matrices, multiplication

$$=\sum_{k=1}^{m} \sum_{l=1}^{n} a_{kl} b_{lk}$$                distribution

$$=\sum_{l=1}^{n}  \sum_{k=1}^{m}  a_{kl}  b_{lk}$$                 finite sum

$$=\sum_{l=1}^{n}  \sum_{k=1}^{m} b_{lk} a_{kl}  $$                 complex number

$$=\sum_{l=1}^{n}  (\sum_{k=1}^{m} b_{lk} a_{kl} ) $$               distribution

$$=\sum_{l=1}^{n} (BA)_{ll}$$                            def of matrix multiplication

$$=Tr(BA)$$                                   def of $$Tr$$

------

Lemma 2:

$$A$$ dimension is $$ u \times v$$

$$B$$ dimension is $$v \times w$$

$$C$$ dimension is $$w \times r$$

Then $$A \times (B \times C)$$ is $$u \times r$$, $$(A \times B) \times C$$ is $$u \times r$$

say $$M \equiv [m_{ij}]$$, $$N \equiv [n_{ij}]$$

$$m_{ij}=n_{ij}$$

$$m_{ij}=(A(BC))_{ij}$$

$$=\sum_{k=1}^{v} a_{ik}(BC)_{kj}$$                  def of matrix multiplication

$$=\sum_{k=1}^{v}a_{ik}(\sum_{l=1}^{w}b_{kl}c_{lj})$$        def of matrix multiplication

where $$(\sum_{l=1}^{w}b_{kl}c_{lj})=(BC)_{kj}$$

$$=\sum_{k=1}^{v} \sum_{l=1}^{w} a_{ik} b_{kl} c_{lj}$$            distribution

$$= \sum_{l=1}^{w} (\sum_{k=1}^{v} a_{ik} b_{kl}) c_{lj}$$          finite sum

where $$(\sum_{k=1}^{v} a_{ik} b_{kl})=(AB)_{il}$$

$$= \sum_{l=1}^{w} (AB)_{il} c_{lj}$$                       def of matrix multiplication

$$=((AB)C)_{ij}$$                               def of matrix multiplication

$$=n_{ij}$$                                              

------

$$Tr(A_1 \: A_2\: ... \: A_{n-1}\: A_n)=Tr((A_1 \: A_2\: ... \: A_{n-1})\: A_n)$$

$$=Tr(A_n \:(A_1 \: A_2\: ... \: A_{n-1}))$$

$$=Tr(A_n \: A_1\: ... \: A_{n-1})$$

------

| $$A$$          | $$B$$          | $$A+B$$        |
| -------------- | -------------- | -------------- |
| $$n \times n$$ | $$n \times n$$ | $$n \times n$$ |
| diagonal       | diagonal       | diagonal       |
| triangular     | triangular     | triangular     |
| upper          | upper          | upper          |
| lower          | lower          | lower          |

Invertible Matrices

$$A$$ is invertible if and only if $$\exists B: AB=BA=I_n$$

$$I=\begin{bmatrix}
1 & 0 & ... & 0 \\ ... &...&...&...\\ 0 & 0 & ... & 1\\\end{bmatrix}$$

Properties:

1. $$A^{-1}A=I_n$$
2. $$(AB)^{-1}=B^{-1}A^{-1}$$
3. $$(A^T)^{-1}=(A^{-1})^T$$

------

$$A \in \C^{m \times n}$$

Hermitian

$$A^{H}=(A^{\ast})^T=(A^T)^{\ast}$$

If $$A \in \R^{m \times n}$$ , $$A^{H}=(A^{\ast})^T=(A^T)^{\ast}$$

Normal Matrices

$$A^TA=AA^H$$

Complex: 

- Hermitian matrices: $$A=A^H$$
- Skew Hermitian: $$A=-A^H$$
- Unitary: $$A^{-1}=A^H$$

Real:

- Symmetric: $$A=A^T$$
- Skew symmetric: $$A=-A^T$$
- Orthogonal: $$A^{-1}=A^T$$

------

Block Matrices

$$A=\begin{bmatrix}
a_{11} & a_{12} & a_{13} & a_{14} \\ a_{21} &a_{22}&a_{23}&a_{24}\\ a_{31} &a_{32}&a_{33}&a_{34}\\a_{41} &a_{42}&a_{43}&a_{44}\\\end{bmatrix}$$

## Week 2 Session 1

### Outlines

Linear System: Lines, Hyperplane, Normal

Equivalent Systems: Elementary row operations

Echelon Form: Gaussian Elimination

Row Canonical Form: Gauss-Jordan

------

### Located Vectors

$$P(u_1,..., u_n)$$

$$Q(v_1,...,v_n)$$

![image-20240915111553495](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240915111553495.png)

$$\overrightarrow{\rm PQ}=\overrightarrow{\rm Q}-\overrightarrow{\rm P}=\begin{bmatrix}v_1\\...\\v_n\end{bmatrix}-\begin{bmatrix}u_1\\...\\u_n\end{bmatrix}=\begin{bmatrix}v_1-u_1\\...\\v_n-u_n\end{bmatrix}$$

### Lines

![image-20240915111710513](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240915111710513.png)

$$L=\{x \in \R^{n}: x=p+ut, t\in \R^n\}$$

$$L$$ is a line that passes through point $$P$$ with direction $$u\in\R^n$$

### Linear Systems

#### Linear Equation

$$a_1x_1+...+a_nx_n=b$$

$$\sum_{j=1}^{n}a_jx_j=b$$

where $$a_j$$ are the coefficients, and $$x_j$$ are the unknowns



Hyperplane $$H$$: 

$$H=\{x \in \R^n: \sum_{j=1}^{n}a_jx_j=b\}$$

Example:

$$6x=6$$ , $$H=\{1\}$$

$$x+y=2$$ 

![image-20240915111959698](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240915111959698.png)

$$x+y+z=1$$

Normal to $$H$$: $$\sum_{j=1}^{n}a_jx_j=b$$

$$w \in \R^n$$ such that for all any located vector $$\overrightarrow{\rm PQ}$$ in $$H$$, $$w$$ is orthogonal to $$\overrightarrow{\rm PQ}$$

$$w=\begin{bmatrix}a_1\\...\\a_n\end{bmatrix}$$

Proof:

$$\sum_{j=1}^{n}a_jx_j=b$$

$$P(u_1,...,u_n) \in H \implies \sum_{j=1}^{n}a_ju_j=b$$

$$Q(v_1,...,v_n) \in H \implies \sum_{j=1}^{n}a_jv_j=b$$

$$w \perp \overrightarrow{\rm PQ}$$

$$w=\begin{bmatrix}a_1\\...\\a_n\end{bmatrix}$$

$$w \cdot \overrightarrow{\rm PQ}=\begin{bmatrix}a_1\\...\\a_n\end{bmatrix} \cdot \begin{bmatrix}v_1-u_1\\...\\v_n-u_n\end{bmatrix}$$

$$=\sum_{j=1}^{n}a_j(v_j-u_j)$$

$$=\sum_{j=1}^{n}a_jv_j-\sum_{j=1}^{n}a_ju_j$$

$$=b-b$$

$$=0$$

#### Linear Systems

A list of linear equations with the same unknowns

$$m$$ equations and $$n$$ unkowns

$$a_{11}x_1+a_{12}x_2+...+a_{1n}x_n=b_1$$

$$a_{21}x_1+a_{22}x_2+...+a_{2n}x_n=b_2$$

$$...$$

$$a_{m1}x_1+a_{m2}x_2+...+a_{mn}x_n=b_m$$

- Unique solution
- Infinite solution
- No solution

| $$A$$          | $$x$$           | $$b$$          |
| -------------- | --------------- | -------------- |
| $$m \times n$$ | $$ n \times 1$$ | $$m \times 1$$ |

$$A= \begin{bmatrix}
a_{11} & a_{12} & ... & a_{1n} \\ ... &...&...&...\\ a_{m1} & a_{m2} & ... & a_{mn}\\\end{bmatrix}$$

$$x=\begin{bmatrix}x_1\\...\\x_n\end{bmatrix}$$

$$b=\begin{bmatrix}b_1\\...\\b_m\end{bmatrix}$$

Degenerate linear equation:

$$0x_1+...+0x_n=b$$

1. $$b=0$$, every $$x\in\R^n$$ is a solution 
2. $$b\neq0$$, no solution

Homogenous system: $$Ax=b=\mathbf{0}$$

------

### Equivalent Systems

$$Ax=b$$, $$A'x=b'$$ where $$x$$ is in dimension $$n \times 1$$

Theorem:

Let $$L$$ be a linear combination of the equations $$m$$ $$Ax=b$$,, then $$x$$ is a solution to $$L$$
Proof:

$$Ax=b$$

$$\sum_{j=1}^{n}a_{ij}x{j}=b_i$$ where $$1 \leq v  \leq m$$

Let $$s=\begin{bmatrix}s_1\\...\\s_n\end{bmatrix}$$ is a solution to $$Ax=b$$

Then: $$\sum_{j} \sum_{j=1}^{n}a_{ij}x{j} = \sum_{j} b_i$$          Integration

$$\sum_{i=1}^{m}c_i(\sum_{j=1}^{n}a_{ij}s_j)=\sum_{i=1}^{m} \sum_{j=1}^{n}c_ia_{ij}s_j$$

$$=\sum_{j=1}^{n}(\sum_{i=1}^{m}c_ia_{ij})s_j$$

$$=\sum_{j=1}^{m}c_ib_i$$

$$x$$ is also a solution to $$L$$

$$Ax=b$$ Linear combination $$\rightarrow A'x=b'$$



Elementary Row Operations

1. Row swap: $$R_i \leftrightarrow R_j$$
2. Scalar multiplication: $$R_i \rightarrow kR_i$$
3. Sum of a row with a scalar multiple of another row: $$R_i \rightarrow R_i+kR_j$$

Thm:

$$Ax=b$$ and $$A'x=b'$$ where $$A'$$ ($$b'$$) is obtained form the elementary row operations on $$Ax=b$$ then they have same solutions.

### Geometry: Linear System Solutions

$$Ax=b$$

Row:

$$\sum_{j=1}^{n}a_{ij}x_{j}=b_i$$

Row 1: $$a_{11}x_1+a_{12}x_2+...+a_{1n}x_n=b_1$$

Row 2: $$a_{21}x_1+a_{22}x_2+...+a_{2n}x_n=b_2$$

$$...$$

Row m: $$a_{m1}x_1+a_{m2}x_2+...+a_{mn}x_n=b_m$$



Example 1:

$$x+y=2$$

$$x-y=0$$

$$x=1, y=1$$ is the unique solution

![image-20240915122407177](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240915122407177.png)

Example 2:

$$x+y=2$$

$$2x+2y=4$$

Infinite solution

![image-20240915122442360](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240915122442360.png)

Example 3:

$$x+y=2$$

$$x+y=0$$

No solution

![image-20240915122508376](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240915122508376.png)

------

### Column

$$Ax=b$$

$$A= \begin{bmatrix}
... & ... & ... & ... \\ a_{11} & a_{i2} & ... & a_{in}\\ ... & ... & ... & ...\\\end{bmatrix}$$

$$x=\begin{bmatrix}x_1\\...\\x_n\end{bmatrix}$$

$$b=\begin{bmatrix}b_1\\...\\b_m\end{bmatrix}$$

$$\sum_{j=1}^{n}A_{ij}x_j=b$$



Example1:
$$x+y=2$$

$$x-y=0$$

$$\begin{bmatrix}1 & 1\\1 & -1\end{bmatrix} \begin{bmatrix}x\\y\end{bmatrix}=\begin{bmatrix}2\\0\end{bmatrix}$$

![image-20240915122853311](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240915122853311.png)

Example 2:

$$x+y=2$$

$$2x+2y=4$$

$$\begin{bmatrix}1 & 1\\2 & 2\end{bmatrix} \begin{bmatrix}x\\y\end{bmatrix}=\begin{bmatrix}2\\4\end{bmatrix}$$

![image-20240915122944114](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240915122944114.png)

------

### Echelon Form

$$2x_1+3x_2+x_3+x_4-x_5=2$$

$$x_2+x_3+0x_4+x_5=2$$

$$x_4+x_5=1$$

$$m=3$$, $$n=5$$

Pivot variables: $$x_1,x_2,x_4$$ (leading variables)

Free variables: $$x_3,x_5$$ (non-leading variables)



Special case (Triangular Form)

$$2x_1+3x_2+4x_3=5$$

$$2x_2+x_3=6$$

$$3x_3=1$$

$$m=3, n=3$$

------

### Gaussian Elimination

Two step process for solving linear systems of form $$Ax=b$$

1. Forward elimination: Reduce to Echelon Form
2. Backward substitution



Example 1:

$$R1: 2x+y+z=5$$

$$R2:4x-6y=-2$$

$$R3: -2x+7y+2z=9$$



Forward Elimination:

$$R1: R1$$

$$R2: R2-2R1$$

$$R3: R3+R1$$

$$2x+y+z=5$$

$$0x-8y-2z=-12$$

$$0x+8y+3z=14$$



$$R1: R1$$

$$R2: R2$$

$$R3: R3+R2$$

$$2x+y+z=5$$

$$0x-8y-2z=-12$$

$$0x+0y+z=2$$



Backward Substitution:

$$z=2$$

$$y=1$$

$$x=1$$



Augmented Matrix ($$M$$)

| $$A$$          | $$x$$          | $$b$$          | $$M$$              |
| -------------- | -------------- | -------------- | ------------------ |
| $$m \times n$$ | $$n \times 1$$ | $$m \times 1$$ | $$m \times (n+1)$$ |

$$M \equiv \begin{bmatrix}A \:|\: b\\\end{bmatrix}$$

$$M=\begin{bmatrix}2 & 1 & 1 \:|\: 5 \\4 & -6 & 0 \:|\: -2 \\ -2 & 7 & 2 \:|\: 9\end{bmatrix}$$

Where $$A=\begin{bmatrix}2 & 1 & 1\\4 & -6 & 0\\-2 & 7 & 2\end{bmatrix}$$, $$b=\begin{bmatrix}5\\-2\\9\end{bmatrix}$$

Echelon Matrix:

$$M=\begin{bmatrix}2 & 1 & 2 & 1 \\0 & 4 & 3 &2 \\ 0 & 0 & 2 &1\\0 & 0 & 0 &5\end{bmatrix}$$

## Week 2 Session 2

### Outline

Row Canonical Form: Gauss Jordan Elimination

Elementary Matrix Operations

LU Decomposition: LDU

Vector Spaces

------

<u>Echelon Matrix</u>

$$\begin{bmatrix}1 & 1 & 2 & 3 &5\\0 & 2 & 1 & 4 & -1\\0&0&0&2&1\end{bmatrix}$$

<u>Augmented Matrix</u>

$$Ax=b$$, $$M=[A|b]$$

### Row Canonical Form (Row-reduced Echelon Form)

1. Echelon Form
2. All non zero leading elements must be equal to 1
3. All the other values above and below a leading element must be 0

$$\begin{bmatrix}1 & 0 & 3 & 0 &1\\0 & 1 & 2 & 0 & 2\\0&0&0&1&2\end{bmatrix}$$

$$M=[A|b]$$

### Gauss-Jordan Elimination

$$Ax=b$$

$$M=[A|b]$$ - Augmented matrix

Reduce $$M$$ to its row canonical form

$$M'=[A'|b']$$ (i.e., $$A'x=b'$$)



Example: 

$$2x_1+x_2+x_3=5$$

$$4x_1-6x_2=-2$$

$$-2x_1+7x_2+2x_3=9$$

$$A=\begin{bmatrix}2 & 1 & 1\\4 & -6 & 0\\-2 & 7 & 2\end{bmatrix}$$

$$b=\begin{bmatrix}5\\-2\\9\end{bmatrix}$$

$$M \equiv [A|b] = \begin{bmatrix}2 & 1 & 1 |5\\4 & -6 & 0|-2\\-2 & 7 & 2|9\end{bmatrix}$$



$$R1:R1$$

$$R2:R2-2R1$$

$$R3:R3+R1$$

$$\begin{bmatrix}2 & 1 & 1 &5\\0 & 8 & -2&-12\\0 & 8 & 3&14\end{bmatrix}$$



$$R1:R1$$

$$R2:R2$$

$$R3:R3+R2$$

$$\begin{bmatrix}2 & 1 & 1 &5\\0 & 8 & -2&-12\\0 & 0 & 1&2\end{bmatrix}$$ which is the Echelon Form



$$R1:R1-R3$$

$$R2:R2+2R3$$

$$R3:R3$$

$$\begin{bmatrix}2 & 1 & 0 &3\\0 & -8 & 0&-8\\0 & 0 & 1&2\end{bmatrix}$$



$$R1:R1$$

$$R2:-1/8R2$$

$$R3:R3$$

$$\begin{bmatrix}2 & 1 & 0 &3\\0 & 1 & 0&1\\0 & 0 & 1&2\end{bmatrix}$$



$$R1:R1-R2$$

$$R2:R2$$

$$R3:R3$$

$$\begin{bmatrix}2 & 0 & 0 &2\\0 & 1 & 0&1\\0 & 0 & 1&2\end{bmatrix}$$



$$R1:1/2R1$$

$$R2:R2$$

$$R3:R3$$

$$\begin{bmatrix}1 & 0 & 0 &1\\0 & 1 & 0&1\\0 & 0 & 1&2\end{bmatrix}$$, which is in row canonical form

$$x_1=1, x_2=2, x_3=2$$

------

### Linear combination of orthogonal vectors

Let $$u_1, u_2, ..., u_n \in\R^n$$ are mutually orthogonal

For any vector $$v \in \R$$

$$v=u_1x_1+...+u_nx_n$$

where $$x_i= \frac{v \cdot u_1}{||u_i||^2}$$ and $$u_i \neq \mathbf{0}$$ for $$1 \leq i \leq n$$

 $$A= \begin{bmatrix}
... & ... & ... & ... \\ u_1 & u_2 & ... & u_n\\ ... & ... & ... & ...\\\end{bmatrix}$$

$$Ax=v$$ what is $$x$$ ?

Proof:

$$u_i \cdot u_j=\begin{cases}0, & \text{if}{\: i \neq j}\\||u_i||^2, & \text{if}{\:i=j}\end{cases}$$     Equation 1

$$Ax=v$$

$$\sum_{j=1}^{n}x_ju_j=v$$                             Equation 2

$$v \cdot u_i=\sum_{j=1}^{n}x_ju_j \cdot u_i$$

$$=\sum_{j=1}^{n}x_j(u_j \cdot u_i)$$

$$=(u_i \cdot u_i)x_i+ \sum_{j=1, j\neq i}^{n}x_j(u_i \cdot u_j)$$

$$=||u_i||^2x_i$$

Therefore, $$v \cdot u_i=||u_i||^2x_i$$ means that $$x_i= \frac{v \cdot u_1}{||u_i||^2}$$ 

$$v=\sum_{j=1}^{n}x_iu_i=\sum_{j=1}^{n}\frac{v \cdot u_i}{||u_j||^2}u_i$$

------

### Inverse Matrix

Using Gauss Jordan Elimination for $$A^{-1}$$

If $$A$$ ($$n \times n$$) is invertible, $$\exists A^{-1}$$ such that $$AA^{-1}=I$$ 

$$AA^{-1}=I$$

say $$B=A^{-1}$$

$$A= \begin{bmatrix}
... & ... & ... & ... \\ a_1 & a_2 & ... & a_n\\ ... & ... & ... & ...\\\end{bmatrix}$$

$$A= \begin{bmatrix}
... & ... & ... & ... \\ b_1 & b_2 & ... & b_n\\ ... & ... & ... & ...\\\end{bmatrix}$$

$$\begin{bmatrix}
... & ... & ... & ... \\ a_1 & a_2 & ... & a_n\\ ... & ... & ... & ...\\\end{bmatrix}\begin{bmatrix}
... & ... & ... & ... \\ b_1 & b_2 & ... & b_n\\ ... & ... & ... & ...\\\end{bmatrix}=\begin{bmatrix}
1 & 0 & 0 & 0 \\ ... & ... & ... & ...\\ 0 & 0 & ... & 1\\\end{bmatrix}$$

$$Ab_1=\begin{bmatrix}1 \\ 0\\ ... \\0 \\\end{bmatrix}$$

$$Ab_2=\begin{bmatrix}0 \\ 1\\ ... \\0 \\\end{bmatrix}$$

$$M=[A|I]$$ Row canonical $$\rightarrow$$ [$$I|A^{-1}]$$



Example 1:

$$A= \begin{bmatrix}
2 & 1 & 1 \\ 4 & -6 & 0\\ -2 & 7 & 2\\\end{bmatrix}$$  Find $$A^{-1}$$

$$M \equiv \begin{bmatrix}2 & 1 & 1 &1& 0&0\\ 4 & -6 & 0&0&1&0\\ -2 & 7 & 2&0&0&1\\\end{bmatrix}$$

$$R1:R1$$

$$R2:R2$$

$$R3=R3+R2$$

$$\begin{bmatrix}2 & 1 & 1 &1& 0&0\\ 0 & -8 & -2&-2&1&0\\ 0 & 8 & 3&1&0&1\\\end{bmatrix}$$



$$R1:R1-R3$$

$$R2:R2$$

$$R3:R3+R2$$

$$\begin{bmatrix}2 & 1 & 1 &1& 0&0\\ 0 & -8 & -2&-2&1&0\\ 0 & 0 & 1&-1&1&1\\\end{bmatrix}$$



$$R1:R1-R3$$

$$R2:R2+2R3$$

$$R3:R3$$

$$\begin{bmatrix}2 & 1 & 0 &2& -1&-1\\ 0 & -8 & 0&-4&3&2\\ 0 & 0 & 1&-1&1&1\\\end{bmatrix}$$



$$R1:R1$$

$$R2=-1/8R2$$

$$R3=R3$$

$$\begin{bmatrix}2 & 1 & 0 &2& -1&-1\\ 0 & 1 & 0&1/2&-3/8&-1/4\\ 0 & 0 & 1&-1&1&1\\\end{bmatrix}$$



$$R1:R1-R2$$

$$R2:R2$$

$$R3=R3$$

$$\begin{bmatrix}2 & 0 & 0 &3/2& -5/8&-3/4\\ 0 & 1 & 0&1/2&-3/8&-1/4\\ 0 & 0 & 1&-1&1&1\\\end{bmatrix}$$



$$R1:1/2R1$$

$$R2:R2$$

$$R3:R3$$

$$\begin{bmatrix}1 & 0 & 0 &3/4& -5/16&-3/8\\ 0 & 1 & 0&1/2&-3/8&-1/4\\ 0 & 0 & 1&-1&1&1\\\end{bmatrix}$$

where $$A^{-1}=\begin{bmatrix}3/4& -5/16&-3/8\\ 1/2&-3/8&-1/4\\ -1&1&1\\\end{bmatrix}$$

Check:

$$AA^{-1}=\begin{bmatrix}
2 & 1 & 1 \\ 4 & -6 & 0\\ -2 & 7 & 2\\\end{bmatrix} \begin{bmatrix}3/4& -5/16&-3/8\\ 1/2&-3/8&-1/4\\ -1&1&1\\\end{bmatrix}=\begin{bmatrix}1& 0&0\\ 0&1&0\\ 0&0&1\\\end{bmatrix}$$

------

### Elementary Matrix Operations

$$eA \equiv EA$$

where $$e$$ is the elementary row operation, $$E$$ is the elementary matrix operation

$$e_n...e_1A=E_n...E_1A$$

1. Row Swap $$R_i \leftrightarrow R_j$$

   ![image-20240915173707668](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240915173707668.png)

   $$EA=\begin{bmatrix}1& 0&0\\ 0&0&1\\ 0&1&0\\\end{bmatrix}A=\begin{bmatrix}...b_1:\\ ...b_2:\\ ...b_3:\\\end{bmatrix}$$

   Let $$E=I$$

   $$I=\begin{bmatrix}1& 0&0\\ 0&1&0\\ 0&0&1\\\end{bmatrix}$$

   $$EA \equiv B$$ where $$B=[b_{ij}]$$

   $$\sum_{k=1}^{n}e_{ik}a_{kj}=b_{ij}$$

   where $$e_{ik}=[e_{i1}, e_{i2}, ..., e_{in}]$$

   ![image-20240915174121736](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240915174121736.png)

   $$\begin{bmatrix}1& 0&0\\ 0&0&1\\ 0&1&0\\\end{bmatrix} \begin{bmatrix}1& 0&0\\ 0&0&1\\ 0&1&0\\\end{bmatrix}=\begin{bmatrix}1& 0&0\\ 0&1&0\\ 0&0&1\\\end{bmatrix}$$

2. Scalar Multiplication of a row

   $$R_i:kR_i$$

   $$EA=B$$

   $$R1:R1$$

   $$R2:kR2$$

   $$R3:R3$$

   $$E=\begin{bmatrix}1& 0&0\\ 0&k&0\\ 0&0&1\\\end{bmatrix}$$ and $$E^{-1}=\begin{bmatrix}1& 0&0\\ 0&1/k&0\\ 0&0&1\\\end{bmatrix}$$

3. Row addition with a scalar multiple of another row

   | Operation | $$E$$      | $$E^{-1}$$     |
   | --------- | ---------- | -------------- |
   | $$R1$$    | $$R1$$     | $$R1$$         |
   | $$R2$$    | $$R2+kR3$$ | $$R2+kR3-kR3$$ |
   | $$R3$$    | $$R3$$     | $$R3$$         |

   This is an operation of $$E$$ and $$E^{-1}$$

   $$E=\begin{bmatrix}1& 0&0\\ 0&1&k\\ 0&0&1\\\end{bmatrix}$$ and $$E^{-1}=\begin{bmatrix}1& 0&0\\ 0&1&-k\\ 0&0&1\\\end{bmatrix}$$

------

### LU decomposition

$$A=LU \equiv LDU$$

where $$A$$ is in dimension $$n \times n$$, $$L$$ is the lower triangular, $$U$$ is the upper triangular, $$D$$ is the diagonal matrix

$$A$$ is a nonsingular matrix that can be reduced into triangular from $$U$$ only row-addition operations

Example:

$$A= \begin{bmatrix}
2 & 1 & 1 \\ 4 & -6 & 0\\ -2 & 7 & 2\\\end{bmatrix}$$

$$e_n...e_1A=U=E_n...E_1A$$

$$E_n...E_1A=U$$

$$(E_n...E_1)^{-1}=E_1^{-1}E_2^{-1}...E_n^{-1}$$

$$(E_n...E_1)^{-1}(E_n...E_1)A=E_1^{-1}E_2^{-1}...E_n^{-1}U$$

$$LHS: A=LU$$

$$RHS=LU$$



$$R1:R1$$

$$R2:R2-2R1$$

$$R3:R3+R1$$

$$\begin{bmatrix}
2 & 1 & 1 \\ 0 & -8 & -2\\ 0 & 8 & 3\\\end{bmatrix}$$

$$E_1= \begin{bmatrix}
1 & 0 & 0 \\ -2 & 1 & 0\\ 1 & 0 & 1\\\end{bmatrix}$$

| Operations | $$E_1$$             | $$E_1^{-1}$$ |
| ---------- | ------------------- | ------------ |
| $$R1$$     | $$R1$$              | $$R1$$       |
| $$R2$$     | $R2-2R1$$ ($$+2R1$) | $$R2$$       |
| $$R3$$     | $$R3+R1$$ ($$-R1$$) | $$R3$$       |

$$E_1^{-1}=\begin{bmatrix}
1 & 0 & 0 \\ 2 & 1 & 0\\ -1 & 0 & 1\\\end{bmatrix}$$



$$R1:R1$$

$$R2:R2$$

$$R3:R3+R2$$

$$\begin{bmatrix}
2 & 1 & 1 \\ 0 & -8 & -2\\ 0 & 0 & 1\\\end{bmatrix}$$

$$E_2= \begin{bmatrix}
1 & 0 & 0 \\ 0 & 1 & 0\\ 0 & 1 & 1\\\end{bmatrix}$$

$$(E_2E1)A=U$$

$$A=(E_1^{-1}E_2^{-1})U$$ and  $$ E_1^{-1}E_2^{-1}=L$$

| Operations | $$E_1$$             | $$E_1^{-1}$$ |
| ---------- | ------------------- | ------------ |
| $$R1$$     | $$R1$$              | $$R1$$       |
| $$R2$$     | $$R2$$              | $$R2$$       |
| $$R3$$     | $$R3+R2$$ ($$-R2$$) | $$R3$$       |

$$E_2^{-1}=\begin{bmatrix}
1 & 0 & 0 \\ 0 & 1 & 0\\ 0 & -1 & 1\\\end{bmatrix}$$

$$L=E_1^{-1}E_2^{-1}=\begin{bmatrix}
1 & 0 & 0 \\ 2 & 1 & 0\\ -1 & 0 & 1\\\end{bmatrix}\begin{bmatrix}
1 & 0 & 0 \\ 0 & 1 & 0\\ 0 & -1 & 1\\\end{bmatrix}=\begin{bmatrix}
1 & 0 & 0 \\ 2 & 1 & 0\\ -1 & -1 & 1\\\end{bmatrix}$$

Check:

$$LU=\begin{bmatrix}
1 & 0 & 0 \\ 2 & 1 & 0\\ -1 & -1 & 1\\\end{bmatrix}\begin{bmatrix}
2 & 1 & 1 \\ 0 & -8 & -2\\ 0 & 0 & 1\\\end{bmatrix}=\begin{bmatrix}
2 & 1 & 1 \\ 4 & -6 & 0\\ -2 & 7 & 2\\\end{bmatrix}$$

## Week 3 Session 1

### Outlines

LU Decomposition: LDU

Vector Spaces: Fields, Span, Subspaces

Linear Independence: Invertibility

Uniqueness Theorem

------

$$A=\begin{bmatrix}
2 & 1 & 1 \\ 4 & -6 & 0\\ 2 & 7 & 2\\\end{bmatrix}=\begin{bmatrix}
1 & 0 & 0 \\ 2 & 1 & 0\\ -1 & -1 & 1\\\end{bmatrix}\begin{bmatrix}
2 & 1 & 1 \\ 0 & -8 & -2\\ 0 & 0 & 1\\\end{bmatrix}$$

$$A=LU$$

$$A=LDU$$

$$A=\begin{bmatrix}
1 & 0 & 0 \\ 2 & 1 & 0\\ -1 & -1 & 1\\\end{bmatrix}\begin{bmatrix}
2 & 0 & 0 \\ 0 & -8 & 0\\ 0 & 0 & 1\\\end{bmatrix} \begin{bmatrix}
1 & 1/2 & 1/2 \\ 0 & 1 & 1/4\\ 0 & 0 & 1\\\end{bmatrix}$$

------

### Vector Spaces

Field:

A field $$F$$ is a collection of elements such that for binary operations: $$+, \times$$ 

We have the following: $$\forall a, b, c \in F$$

1. $$a+b=b+a$$ ; $$a \cdot b= b \cdot a$$

2. $$a+(b+c)=(a+b)+c$$ ; $$a \cdot (b \cdot c)=(a \cdot b)\cdot c$$

3. $$\exists 0 \in F$$ : $$a+0=a$$

   $$\exists 1 \in F$$ : $$a \cdot 1=a$$

4. $$\exists a' \in F$$: $$a+a'=0$$
5. $$a \times \frac{1}{a}=1$$ if $$a \neq 0$$
6. $$a \cdot (b+c)=(a \cdot b)+(a \cdot c)$$

Example: 

$$\R, \Q, \C$$ - field

$$\Z$$ not a field ($$5^{th}$$$\frac{1}{a} \notin \Z$ )



A vector $$V$$ over field $$F$$ is a collection of elements $$\{\alpha, \beta, \gamma, ...\}$$ (typically called vectors) and collection of elements $$\{a,b,c,...\} \in F$$ ca;;ed scalars such that:

- Commutative group for ($$V, +$$)
  1. $$\alpha + \beta \in V$$
  2. $$ \alpha + \beta = \beta + \alpha$$
  3. $$\alpha + (\beta + \gamma) =(\alpha + \beta)+ \gamma$$
  4. $$\forall \alpha, \exists \alpha' \in V$$ : $$\alpha + \alpha'=\mathbf{0}$$
  5. $$\exists \mathbf{0}\in V: \forall \alpha \in V, 0+\alpha=\alpha$$

- Properties for combination of $$+$$ and $$\times$$
  1. $$a \alpha \in V$$
  2. $$a(b \alpha)=(ab) \alpha$$
  3. $$a(\alpha+\beta)=a\alpha+a\beta$$
  4. $$(a+b)\alpha=a\alpha+b\alpha$$
  5. $$\exists 1 \in F: 1 \alpha=\alpha$$

:one: 

$$K$$ is field, $$K^n$$

$$\alpha, \beta \in K^n$$

$$\alpha=\begin{bmatrix}
a_1\\...\\a_n\end{bmatrix}$$, $$a_1 \in K$$



:two:

Polynomial Space: $$P(t)$$

$$p(t) \in P(t)$$

$$p(t)=a_0+a_1t^1+a_2t^2+...+a_st^s$$

where $$\ s\in \{1,2,3, ...\} $$



:three:

Matrix over a field: $$K_{m \times n}$$

$$A\in K_{m \times n}$$
$$A \equiv [a_{ij}]$$ where $$a_{ij}\in K$$

------

### Linear Combination:

Let $$\alpha_1, \alpha_2, ... \alpha_n \in V$$ where is a vector space over field $$V$$

$$w$$ is a linear combination of the $$\alpha_i$$'s if:

$$w=a_1\alpha_1+...+a_n\alpha_n$$

where $$a_1, a_2, ..., a_n \in F$$

Alternatively: 

$$Ax=b$$

$$\begin{bmatrix}...\\\alpha_1\\...\end{bmatrix}x_1+\begin{bmatrix}...\\\alpha_2\\...\end{bmatrix}x_2+...+\begin{bmatrix}...\\\alpha_n\\...\end{bmatrix}x_n=w$$

### Linear Span

Let $$S=\{\alpha_1,...\alpha_n\} \subset V$$ for a vector space $$V$$ over field $$F$$

$$S$$ spans $$V$$ means that $$\forall w \in V, \exists a_1,...,a_n \in F$$ such that:

$$w=a_1\alpha_1+...+a_n\alpha_n$$

### Subspace

$$u$$ is a subspace of vector space $$V$$ over field $$F$$, if 

1. $$u\subset V$$ ($$u$$ is a subset of $$V$$)
2. $$u$$ is a vector space over $$F$$

Thm:

Let $$V$$be a vector space over field $$F$$ and $$u$$ is a subset of $$v$$ ($$ u \subset V$$), If: 

1. $$\mathbf{0} \in u$$
2. $$\forall \alpha, \beta \in u, \forall a, b \in F$$, $$a\alpha+b \beta \in u$$

Then $$u$$ is a space of $$V$$



Thm:

Let $$V$$ be a vector space over field $$F$$. If $$u$$ is a subspace of $$V$$, and $$w$$ is  a subspace of $$u$$, then $$w$$ is a subspace of $$V$$



Thm:

Intersection of any number of subspaces of a vector $$V$$ over field $$F$$ is a subspace of $$V$$

Proof:

$$u_1, u_2, ...$$ are subspaces of $$V$$
$$u_1$$ is  a subspace of $$V$$

$$u_2$$ is  a subspace of $$

$$...$$

If $$\bigcap_{i=1}^{n} u_i$$ a subspace of $$V$$? 

Yes.



Example:

$$w=\begin{bmatrix}1\\2\\3\end{bmatrix}=1 \begin{bmatrix}1\\0\\0\end{bmatrix}+2 \begin{bmatrix}0\\1\\0\end{bmatrix}+3 \begin{bmatrix}0\\0\\1\end{bmatrix}$$

where $$\alpha_1=\begin{bmatrix}1\\0\\0\end{bmatrix}$$ , $$\alpha_2=\begin{bmatrix}0\\1\\0\end{bmatrix}$$ , $$\alpha_3=\begin{bmatrix}0\\0\\1\end{bmatrix}$$ :white_check_mark:

where $$\alpha_1=\begin{bmatrix}1\\0\\0\end{bmatrix}$$ , $$\alpha_2=\begin{bmatrix}0\\1\\1\end{bmatrix}$$ :negative_squared_cross_mark:



$$\R^2 \equiv \{(x,y): x\in \R, y\in \R\}$$

$$\{\mathbf{0}\}$$ subspace of $$\R^2$$ :white_check_mark:

$$ax+by=1$$ :negative_squared_cross_mark:

$$ax+by=0$$ subspace of $$\R^2$$

------

Thm:

Let $$S=\{\alpha_1, ... \alpha_n\} \subset V$$ where $$V$$ is a vector space over $$F$$ and $$L(s)$$ be the set of all linear combinations of $$S$$ with respect to $$F$$. Then $$L(s)$$ is a subspace of $$V$$.

1. Vector space $$V$$ over field $$F$$
2. $$S=\{\alpha_1, ... \alpha_n\} \subset V$$
3. $$L(s)=\{w: w=\sum_{i=1}^{n}a_i\alpha_1, a_i\in F, \alpha_i \in \S\}$$

$$\implies L(s)$$ (span of $$S$$) is a subspace of $$V$$



Proof:

1. Show that $$L(s) \subset V$$

   $$v \in L(s) \implies v \in V$$

   Assume that $$v \in L(s)$$

   $$v=\sum_{i=1}^{n}a_i \alpha_i$$                      - Def of $$L(s)$$

   $$\alpha_1 \in S \implies \alpha_i \in V$$           - because $$S \subset V$$

   $$v=\sum_{i=1}^{n}a_i \alpha_i \in V$$             - $$V$$ is a vector space

   $$L(s) \subset V$$

2. Show that $$\mathbf{0} \in L(s)$$

   $$\mathbf{0}=0\alpha_1+0\alpha_2+...+0\alpha_n=\sum_{i=1}^{n}0\alpha_i \in L(s)$$  - Def of $$S$$

3. Show that for $$v,w \in L(s)$$ and $$c,d \in F$$, $$cv+dw \in L(s)$$

   $$cv+dw=c\sum_{i=1}^{n}a_i \alpha_i+ d\sum_{i=1}^{n} b_i \alpha_i$$ where $$v=\sum_{i=1}^{n}a_i \alpha_i$$ and $$w=\sum_{i=1}^{n} b_i \alpha_i$$

   $$=\sum_{i=1}^{n}ca_i\alpha_i+\sum_{i=1}^{n}db_i\alpha_i$$

   $$=\sum_{i=1}^{n}(ca_i+db_i)\alpha_i$$ where $$ca_1+db_i \in F$$ 

   Therefore, $$cv+dw \in L(s)$$

$$L(s)$$ is a subspace of $$V$$

------

### Linear Independence

Let $$v$$ be a vector space over field $$F$$

$$S=\{\alpha_1, ... \alpha_n\} \subset v$$

$$s$$ is a linearly dependent set if there exist $$a_i$$'s in $$F$$ such that:

$$a_1\alpha_1+a_2\alpha_2,..., a_n\alpha_n=\mathbf{0}$$

and at least one of the $$a_i$$'s is non-zero



Linearly Independent:

$$s$$ is linearly independent means that: 

$$a_1\alpha_1+a_2\alpha_2,..., a_n\alpha_n=\mathbf{0}$$ only holds when:

$$a_1=a_2=...=a_n=0$$

$$Ax=\mathbf{0}$$ - Homogenous System

 $$A= \begin{bmatrix}
... & ... & ... & ... \\ \alpha_1 & \alpha_2 & ... & \alpha_n\\ ... & ... & ... & ...\\\end{bmatrix}$$ , $$x=\begin{bmatrix}x_1\\...\\x_n\end{bmatrix}$$ , $$b=\mathbf{0}$$

Note:

Let $$S=\{\alpha_1, ... \alpha_n\} \subset v$$ ,then: 

1. If $$\mathbf{0} \in s$$, then $$s$$ is a linearly dependent set
2. If $$s=\{\alpha_1\}$$, then $$s$$ is linearly dependent if and only if $$\alpha_1=0$$

------

### Row Equivalence

$$A$$ , $$B$$ are in dimension of $$m \times n$$
$$A$$ is row equivalent to $$B$$ if f$$B$$ can be obtained from a sequence of elementary row operations of $$A$$



Example

$$A$$ row operations $$\implies$$ $$A'$$ (Echelon Form) row operations $$\implies$$ $$A''$$ (Row Canonical Form)

Say $$A$$ in dimension of $$n \times n$$

Echelon Form

$$L=\begin{bmatrix} 1& 2 & 3 & 4 \\ 0 & 2 & 4 & 5\\ 0 & 0 & 3 & 1\\ 0 & 0 & 0 & 1\end{bmatrix}$$, number of pivots (1, 2, 3, 1) $$=n$$

$$R=\begin{bmatrix} 1& 2 & 3 & 4 \\ 0 & 2 & 4 & 5\\ 0 & 0 & 0 & 2\\ 0 & 0 & 0 & 1\end{bmatrix}$$ , number of pivots (1, 2, 2) $$<n$$   Linearly dependent, 0 row (R4)

Row Canonical Form

$$L=\begin{bmatrix} 1& 0 & 0 & 0 \\ 0 & 1 & 0 & 0\\ 0 & 0 & 1 & 0\\ 0 & 0 & 0 & 1\end{bmatrix}=I$$

$$R=\begin{bmatrix} 1& 0 & x & 0 \\ 0 & 1 & y & 0\\ 0 & 0 & 0 & 1\\ 0 & 0 & 0 & 0\end{bmatrix}\neq I$$

$$I^{-1}=I$$

$$A \in \R^{n \times n}=\begin{cases}A \sim (Row\: Equivalent)\: I\\A \nsim I\end{cases}$$



$$B=\begin{bmatrix} ... & ... & ... & ...\\ ... & ... & ... & ...\\ ... & ... & ... & ...\\ 0 & 0 & 0 & 0 \end{bmatrix}$$(zero row) , $$B=\begin{bmatrix} ... & ... & ... & ...\\ ... & ... & ... & ...\\ \beta_1 & \beta_2 & \beta_3 & \beta_4\\ ... & ... & ... & ... \end{bmatrix}$$

$$BB^{-1}\neq \begin{bmatrix} 1& 0 & 0 & 0 \\ 0 & 1 & 0 & 0\\ 0 & 0 & 1 & 0\\ 0 & 0 & 0 & 1\end{bmatrix}=I$$

There is no $$\beta_4$$ such that $$i_{44}=1$$

So $$B^{-1}$$ does not exist

## Week 3 Session 2

### Outlines

Uniqueness Theorem 

Basis and Dimension: Dimension Theorem

Subspaces of a matrix

------

$$A \in \R^{n \times n}$$ 

1. Linearly independent rows $$\leftrightarrow A \sim I$$
2. Linearly dependent rows $$\leftrightarrow A\sim B$$ such that $$B^{-1}$$ does not exist

Thm:

Let $$A$$ be a square matrix, the following statement are equivalent: 

1. $$A$$ is invertible 
2. $$A$$ is row equivalent to $$I$$
3. $$A$$ is a product of elementary matrices

------

Let $$P$$ and $$Q$$ be logical statements

If $$P$$ then $$Q$$ ($$P \implies Q$$ )

1. Assume $$P$$ is TRUE, and then show it logically implies that $$Q$$ us TRUE
2. Proof by contradiction: $$\sim Q \implies \sim P$$

------

If and only if (Equivalence)

$$P$$ if and only if ($$P \leftrightarrow Q$$)

- $$P \implies Q$$
- $$Q \implies P$$

------

Proof: $$a \implies b$$, $$b \implies c$$, $$c \implies a$$

$$a \implies b$$, $$b \implies c$$, $$c \implies a$$

Then, $$a \leftrightarrow b$$

- $$a \implies b$$

  $$A$$ is invertible $$\implies$$ $$A$$ is row equivalent to $$I$$

  $$P \implies Q$$

  $$\sim Q$$ : If $$A$$ if not row equivalent to $$I$$, then $$A \sim B$$ such that $$B^{-1}$$ does not exist

  so, $$B=E_n...E_1A$$

  $$(E_n...E_1)^{-1}B=(E_n...E_1)^{-1}E_n...E_1A=A$$

  Due to $$(A_1A_2)^{-1}=A_2^{-1}A_1^{-1}$$

  $$A^{-1}=B^{-1}(E_n...E_1)$$

  So, $$A$$ is not invertible because $$B^{-1}$$ does not exist

- $$b \implies c$$

  If $$A$$ is row equivalent to $$I$$ then $$A$$ is a product of elementary matrices

  $$P \implies Q$$

  $$E_n...E_1A=I$$

  $$(E_n...E_1)^{-1}(E_n...E_1)A=(E_n...E_1)^{-1}I$$

  so, $$A=(E_n...E_1)^{-1}=E_1^{-1}...E_n^{-1}$$

  For an elementary matrix $$E_i$$, $$E_i^{-1}$$ is also an elementary matrix

- $$c \implies a$$

  If $$A$$ is a product of elementary matrices then $$A$$ is invertible

  $$A=(E_1...E_n)$$

  $$A^{-1}=(E_1...E_n)^{-1}=E_n^{-1}...E_1^{-1}$$ because $$E_1^{-1}$$ exists

Therefore, $$a \implies b$$ , $$b \implies c$$ , $$c \implies a$$

------

Thm: 

Let $$v$$ be a vector space over $$F$$ and $$S=\{\alpha_1, ..., \alpha_n\} \subset V$$. Suppose $$S$$ is a linearly independent set, then for every $$w \in V$$ there exist at most one representation as a linear combination of vectors in $$S$$. 

Sketch:

If $$S=\{\alpha_1, ..., \alpha_n\} \subset V$$ (linearly independent set), then $$\forall w \in V$$, there exist at least one representation: $$w=\sum_{i=1}^{n}a_i\alpha_i$$

$$P \implies Q$$

Proof:

$$\sim Q$$ : Assume that $$\exists w \in V$$, we have two possible representations

$$w=\sum_{i=1}^{n}a_i\alpha_i$$ and $$w=\sum_{i=1}^{n}b_i\alpha_i$$ , $$\exists k: a_k \neq b_k, 1 \leq k \leq n$$

So, $$\mathbf{0}=w-w= \sum_{i=1}^{n}a_i\alpha_i-\sum_{i=1}^{n}b_i\alpha_i$$

$$=\sum_{i=1}^{n}(a_i-b_i)\alpha_i$$

$$\mathbf(0)=\sum_{i=1, i \neq k}^{n}(a_i-b_i)\alpha_i+(a_k-b_k)\alpha_k$$ , where $$(a_k-b_k) \neq 0$$

Therefore, $$S$$ is linearly dependent set

------

$$S=\{\alpha_1, ..., \alpha_n\} \subset V$$ (vector space over field $$F$$)

If $$S$$ spans $$V$$ then $$\forall w \in v$$, $$\exist a_i$$'s $$\in F: w=\sum_{i=1}^{n}a_i\alpha_i$$ 

$$P \implies Q$$

Thm:

Let $$S=\{\alpha_1, ..., \alpha_n\} \subset V$$ where $$V$$ is a vector space over field $$F$$

If:

1. $$S$$ is linearly independent (number of representations $$\leq 1$$)

2. $$S$$ spans $$V$$ (number of representations $$\geq 1$$) 

then every vector $$w \in V$$ has a unique representation as a linear combination of vectors in $$S$$ 

------

Properties

$$S=\{\alpha_1, ..., \alpha_n\} \subset V$$ (vector space over field $$F$$)

1. If $$S$$ is linearly dependent, then any larger set of vectors containing $$S$$ is linearly dependent
2. If $$S$$ is linearly independent, then any subset of $$S$$ is linearly dependent

------

### Basis and Dimension

Vector space $$V$$ over field $$F$$
Basis of $$V$$ is a set of vectors $$S \in V$$ such that:

1. $$S$$ span $$V$$
2. $$S$$ is a linearly independent set

Dimension of $$V$$ is the number of vectors in the basis of $$V$$

Example:

$$V \equiv \R^3$$

$$V={\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}, x_1, x_2, x_3 \in \R}$$

Basis: $$\{\begin{bmatrix}1\\0\\0\end{bmatrix},\begin{bmatrix}0\\1\\0\end{bmatrix},\begin{bmatrix}0\\0\\1\end{bmatrix}\}$$

$$x_1\begin{bmatrix}1\\0\\0\end{bmatrix}+x_2\begin{bmatrix}0\\1\\0\end{bmatrix}+x_3\begin{bmatrix}0\\0\\1\end{bmatrix}$$

$$\{\begin{bmatrix}1\\0\\0\end{bmatrix},\begin{bmatrix}1\\1\\0\end{bmatrix},\begin{bmatrix}1\\1\\1\end{bmatrix}\}$$ :heavy_check_mark:

$$Dim(\R^3)=3$$

$$P_n(t)=$$ Polynomial of order $$\leq n$$

$$p(t) \in P_n(t)$$

$$p(t)=c_0+c_1t^1+...+c_nt^n$$

Basis$$=\{1,t,t^2, ...t^n\}$$

$$p(t)=\sum_{i=0}^{n}c_0t^n$$

$$Dim(P_n(t))=n+1$$

------

Thm: Dimension Theorem

All basis of a vector space have the same number of vectors

Proof:

If $$T=\{\alpha_1, ... \alpha_n\}$$ (a basis) and $$S=\{\beta_1, ..., \beta_m\}$$ (a basis) then $$n=m$$ 

$$P \implies Q$$

Proof by contradiction:

$$\sim Q: n \neq m$$ $$\rightarrow (n<m)$$ or $$(n>m)$$

Let $$(n<m)$$         - (WithoutLossOfGenerality)

$$T=\{\alpha_1, ... \alpha_n\}$$ 

$$S=\{\beta_1, ..., \beta_n, \beta_{n+1}, ... \beta_m\}$$

$$A=\{\alpha_1, ... \alpha_n\}$$ , $$B= \{\beta_1, ..., \beta_n\}$$

$$B=\begin{bmatrix}... & \beta_1 &... \\ ... & \beta_2 & ... \\ ...&...&...\\...& \beta_n &...\end{bmatrix} \in \R^{n \times p}$$ , $$C=\begin{bmatrix}c_{11} & ... &c_{1n} \\ ... & ... & ... \\ ...&...&...\\c_{1n}& ... &c_{nn}\end{bmatrix} \in \R^{n \times n}$$ , $$A=\begin{bmatrix}... & \alpha_1 &... \\ ... & \alpha_2 & ... \\ ...&...&...\\...& \alpha_n &...\end{bmatrix} \in \R^{n \times p}$$

$$B=CA$$

$$\begin{bmatrix}... & \beta_1 &... \\ ... & \beta_2 & ... \\ ...&...&...\\...& \beta_n &...\end{bmatrix}=\begin{bmatrix}c_{11} & ... &c_{1n} \\ ... & ... & ... \\ ...&...&...\\c_{1n}& ... &c_{nn}\end{bmatrix}\begin{bmatrix}... & \alpha_1 &... \\ ... & \alpha_2 & ... \\ ...&...&...\\...& \alpha_n &...\end{bmatrix}$$

$$\beta_{lj}=\sum_{k=1}^{n}c_{lk} \alpha_{kj}$$         - Matrix Multiplication

$$\beta_l=\sum_{k=1}^{n}c_{lk}\alpha_{k}$$



Lemma 1: 

If $$A$$ and $$B$$ have linearly independent rows then $$C$$ is invertible

$$P \implies Q$$ or $$\sim Q \implies \sim P$$

Note: $$C$$ is invertible $$\leftrightarrow$$ $$C$$ has linearly independent rows

$$\sim Q: C$$ has linearly dependent rows

$$c_l=\sum_{i=1, i\neq l}a_ic_i , c_{lk}=\sum_{i=1, i \neq l}^{n}a_ic_{lk}$$

$$\beta_l=\sum_{k=1}^{n}c_{lk}\alpha_k$$

$$=\sum_{k=1}^{n} \sum_{i=1, i \neq l}^{n}a_ic_{lk}\alpha_k$$

$$=\sum_{i=1, i \neq l}^{n}a_i \sum_{k=1}^{n} c_{lk}\alpha_k$$

$$=\sum_{i=1, i \neq l}^{n} a_i \beta_i$$ 



So for $$B=CA$$ with invertible $$C$$ then, 

$$A=C^{-1}B$$

$$C^{-1}=D \equiv [d_{ij}]$$

$$\alpha_{ij}=\sum_{k=1}^{n}d_{ik}\beta_{kj}$$ 

$$\alpha_i=\sum_{k=1}^{n}d_{ik}\beta_{k}$$

$$T=\{\alpha_1, ... \alpha_n\}$$ is a basis of $$V$$ and $$\beta_{m+1} \in V$$

$$\beta_{n+1}=\sum_{i=1}^{n}e_i\alpha_i$$ for some $$e_i$$'s $$\in F$$

$$\beta_{n+1}=\sum_{i=1}^{n}e_i(\sum_{k=1}^{n}d_{ik}\beta_k)$$

$$=\sum_{i=1}^{n} \sum_{k=1}^{n} e_i d_{ik} \beta_k$$

$$=\sum_{i=1}^{n} (\sum_{k=1}^{n} e_i d_{ik}) \beta_k$$

$$S$$ is linearly dependent set

so $$S$$ is a basis

------

### Fundamental subspace of a matrix

$$A \in \R^{m \times n}$$

$$A=\begin{bmatrix} a_{11} & ... & a_{1n} \\ ... & ... & ... \\ a_{m1} & ... & a_{mn}\end{bmatrix}$$

$$Ax=b$$ where $$x \in \R^{n \times 1}, b \in \R^{m \times 1}$$

$$T: \R^{n \times 1}$$ (Domain) $$\rightarrow$$ $$\R^{m \times 1}$$ (Co-domain)

$$A^T y= d$$ where $$A^T \in \R^{n \times m}, y \in \R^{m \times 1}, d \in \R^{n \times 1}$$

$$T: \R^{m \times 1}$$ (Domain) $$\rightarrow$$ $$\R^{n \times 1}$$ (Co-domain)

1. Column Space: $$C(A)$$

   $$C(A)=\{b \in \R^{m \times 1}: Ax=b, x \in \R^{n \times 1}\}$$

2. Row Space: $$C(A^T)$$

   $$C(A^T)=\{d \in \R^{n \times 1}: A^Ty=d, y \in \R^{m \times 1}\}$$

3. Null Space: $$n(A)$$

   $$n(A)=\{x \in \R^{n \times 1}: Ax=\mathbf{0}\}$$

4. Left Null Space: $$n(A^T)$$

   $$n(A)=\{y \in \R^{m \times 1}: A^Ty=\mathbf{0}\}$$

| Subspaces | Dimension                |
| --------- | ------------------------ |
| Domain    | $$n \equiv$$ order       |
| $$C(A)$$  | $$r \equiv$$ rank        |
| $$n(A)$$  | $$\zeta \equiv$$ nullity |

Fact: $$n=r+\zeta$$

$$r \equiv$$ rank

$$r =$$ number of pivots$$=Dim(C(A))=Dim(C(A^T))$$

$$\zeta=$$ number of free variables$$=Dim(n(A))$$

------

Example:

$$A=\begin{bmatrix} 1 & 0\\\ 5 & 4\\ 2 & 4\end{bmatrix}$$ Find $$n(A)$$ and its dimension.

$$Ax=\mathbf{0}$$

 $$\begin{bmatrix} 1 & 0 & 0\\\ 5 & 4 & 0\\ 2 & 4 & 0\end{bmatrix}$$ 



$$R1:R1$$

$$R2:R2-5R1$$

$$R3:R3-2R1$$

$$\begin{bmatrix} 1 & 0 & 0\\\ 0 & 4 & 0\\ 0 & 4 & 0\end{bmatrix}$$ 



$$R1:R1$$

$$R2:R2$$

$$R3:R3-R2$$

$$\begin{bmatrix} 1 & 0 & 0\\\ 0 & 4 & 0\\ 0 & 0 & 0\end{bmatrix}$$ 



$$\begin{bmatrix} 1 & 0\\\ 0 & 4\\ 0 & 0\end{bmatrix} \begin{bmatrix}x\\y\end{bmatrix}=\begin{bmatrix}0\\0\end{bmatrix}$$

$$r=$$ number of pivots $$=2$$$

$$\zeta =$$ number of free variables $$=0$$

$$x=0,y=0$$

$$n(A)=\{\mathbf{0}\}$$

$$Dim(n(A))=\zeta=0$$

------

$$A=\begin{bmatrix} 1 & 0 & 1\\\ 5 & 4 & 9\\ 2 & 4 & 6\end{bmatrix}$$



$$R1:R1$$

$$R2:R2-5R1$$

$$R3:R3-2R1$$

$$\begin{bmatrix} 1 & 0 & 1\\\ 0 & 4 & 4\\ 0 & 4 & 4\end{bmatrix}$$ 



$$R1:R1$$

$$R2:R2$$

$$R3:R3-R2$$

$$\begin{bmatrix} 1 & 0 & 1\\\ 0 & 4 & 4\\ 0 & 0 & 0\end{bmatrix}$$ $$r=2, \zeta=1$$

$$\begin{bmatrix} 1 & 0 & 1\\\ 0 & 4 & 4\\ 0 & 0 & 0\end{bmatrix}\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}$$

Let $$x_3=z, z \in \R$$

$$4x_3+4x_3=0 \implies x_2=-Z$$

$$x_1+x_3=0 \implies x_1=-z$$

$$z\begin{bmatrix}-1\\-1\\1\end{bmatrix}, z \in \R$$

$$n(A)=span\{\begin{bmatrix}-1\\-1\\1\end{bmatrix}\}$$

$$Dim(C(A))=2, Dim(n(A))=1$$



Thm:

Interchanging the rows of a matrix leaves its rank unchanged.

Thm:

If $$Ax=0$$ and $$Bx=0$$ have the same solution, then $$A$$ and $$B$$ have the same column rank

------

## Week 4 Session 1

### Outlines

Dimension Thorem

Existence and Uniqueness

Inner Product Space

------

$$Ax=b$$ where $$A \in \R^{m \times n}, x \in \R^{n \times 1}=y \in \R^{m \times 1}$$ , $$m \neq n$$

Principal Component Analysis (Dimension Reduction)

$$X \in \R^{n \times p}$$  where $$n$$ is the number of sample, $$p$$ is the number of features, $$p \gg 1$$

$$A \in \R^{p \times s}$$ where $$s$$ is a very small dimension

$$X \rightarrow  \bar{X}$$ mean$$=0$$  $$\implies$$ $$K_{xx}$$ where it is $$p \times p$$ $$\rightarrow$$ Eigenvalues

 $$E \in \R^{p \times p}= \begin{bmatrix} ... & ... & ... & ... & .... \\ e_1 & ... & e_s & ... & e_p\\ ... & ... & ... & ... & ...\\(\lambda_{1}) & ... & (\lambda_{s}) & ... & (\lambda_{p}) \end{bmatrix}$$

$$\lambda_1 \gg \lambda_2 \gg ... \gg \lambda_p$$

$$XE$$ where $$X$$ is $$n \times p$$, and $$E$$ is $$p \times p$$

$$X\bar{E}=\hat{X}$$ where $$X$$ is $$n \times p$$, $$\bar{E}$$ is $$p \times s$$, $$\hat{X}$$ is $$n \times s$$

------

Thm:

If $$Ax=0$$ and $$Bx=0$$ have the same solution, then $$A$$ and $$B$$ have the same column rank.

Proof:

$$P \implies Q$$

Let $$s$$ be the column rank of $$A$$

Let $$t$$ be the column rank of $$B$$

where $$t \neq s$$ so, $$(t > s ) / (s>t)$$

Let $$t>s$$ (WLOG)

$$B \in \R^{m \times n}=\begin{bmatrix} ... & ... & ... & ... & .... & ... & ...\\ \beta_1 & ... & \beta_s & ... & \beta_t &... & \beta_n\\ ... & ... & ... & ... & .... & ... & ...\end{bmatrix}$$ so that $$Bx=0$$

where column $$ 1 ... t$$ are linearly independent, $$ t+1 ... n$$ are linearly dependent

$$A \in \R^{m \times n}=\begin{bmatrix} ... & ... & ... & ... & .... & ... & ...\\ \alpha_1 & ... & \alpha_s & ... & \alpha_t &... & \alpha_n\\ ... & ... & ... & ... & .... & ... & ...\end{bmatrix}$$ so that $$Ax=0$$

Therefore $$\exists d_{i}'s \neq 0: \sum_{i=1}^{\sigma}d_i\alpha_i=\mathbf{0}$$ where $$t>s$$

$$\sum_{j=1}^{\sigma}d_j \alpha_j+\sum_{j=t+1}^{n}0 \alpha_j=\mathbf{0}$$

$$x_1=d_1, x_2=d_2, ... x_t=d_t$$ , this is the solution to $$Ax=0$$

$$x_{t+1}=...=x_n=0$$

$$\exists d_{i}'s \neq 0: \sum_{j=1}^{t}d_i \beta_j + \sum_{j=t+1}^{n} 0 \beta_j= \mathbf{0}$$

 $$\exists d_{i}'s \neq 0: \sum_{j=1}^{t}d_i \beta_j=\mathbf{0}$$

$$\{\beta_1, ... \beta_t\}$$ is linearly dependent

Contradiction



Thm:

Elementary row operations preserve column rank 

$$Ax=b$$ elementary operations $$\implies A'x=b'$$ 

$$Ax=0 \implies A'x=0$$



Thm:

### Rank Theorem

Dimension of column space equals the dimension of row space.

$$Ax=b$$ where $$ A \in \R^{m \times n}$$

Proof:

Let $$c$$ be the column rank of $$A$$

Let $$r$$ be the row rank of $$A$$

$$c \leq r$$ or $$r \leq c$$



Case 1: $$c \leq r$$

$$A=\begin{bmatrix}...&\alpha_1&...\\...&...&...\\...&\alpha_r&...\\...&\alpha_{r+1}&...\\...&...&...\\...&\alpha_m&...\end{bmatrix}$$ 

where $$B \in \R^{r \times n}=\begin{bmatrix}...&\alpha_1&...\\...&...&...\\...&\alpha_r&...\end{bmatrix}$$ is linearly independent rows 

where $$D \in \R^{(m-r)\times n}=\begin{bmatrix}...&\alpha_{r+1}&...\\...&...&...\\...&\alpha_m&...\end{bmatrix}$$ is linearly dependent rows

$$\forall_j: r+1 \leq j \leq m, \exists t_{ji}'s:$$

$$\alpha_j=\sum_{i=1}^{r}t_{ji}\alpha_i$$ ,  $$T \equiv [t_{ji}]$$

$$D=TB$$ 

$$ (m-r)\times n = (m-r) \times r (t \times n)$$

$$A=\begin{bmatrix}B\\D\end{bmatrix}=\begin{bmatrix}B\\TB\end{bmatrix}$$

If $$Ax=\mathbf{0} \implies \begin{bmatrix}B\\TB\end{bmatrix}x=\begin{bmatrix}Bx\\TBx\end{bmatrix}=\begin{bmatrix}\mathbf{0}\\\mathbf{0}\end{bmatrix}$$ 

If $$Bx=\mathbf{0} \implies Ax=\begin{bmatrix}B\\TB\end{bmatrix}x=\begin{bmatrix}\mathbf{0}\\\mathbf{0}\end{bmatrix}$$



Case 2: $$r \leq c$$

$$A^Ty=x$$

$$A^T \in \R^{n \times m}, y \in \R^{m \times 1}, x \in \R^{n \times 1}$$

The colmn rank of $$A^T$$ is $$r$$

The row rank of $$A^T$$ is $$c$$

$$A^T=\begin{bmatrix}...&\beta_1&...\\...&...&...\\...&\beta_c&...\\...&\beta_{c+1}&...\\...&...&...\\...&\beta_n&...\end{bmatrix}$$ 

where $$E \in \R^{c \times m}=\begin{bmatrix}...&\beta_1&...\\...&...&...\\...&\beta_c&...\end{bmatrix}$$ is linearly independent rows 

where $$F \in \R^{(n-c)\times m}=\begin{bmatrix}...&\beta_{c+1}&...\\...&...&...\\...&\beta_n&...\end{bmatrix}$$ is linearly dependent rows

$$\forall_j: c+1 \leq j \leq n, \exists r_{ji}'s:$$

$$\beta_j=\sum_{i=1}^{c}r_{ji}\beta_i$$ ,  $$R \equiv [r_{ji}]$$

$$F=RE$$ 

$$A^T=\begin{bmatrix}E\\F\end{bmatrix}=\begin{bmatrix}E\\RE\end{bmatrix}$$

$$A^Ty=\mathbf{0}$$ if and only if $$Ey=0$$

The column rank of $$E=r$$ 

$$Ey=x$$ where $$E \in \R^{c \times m}, y \in \R^{m \times 1}, x \in \R^{c \times 1}$$

$$r \leq c$$

Therefore $$r=c$$



$$Bx=y$$ where $$B \in \R^{m \times n}, x \in \R^{n \times 1}, y \in \R^{m \times 1}$$ 

$$x$$ is the domain and $$y$$ is the co-domain

$$Bx \equiv$$ column smace or range space

column space $$\subset$$ co-domain

------

### Counting Theorem

------

$$A \in \R^{m \times n}$$ 

Dimension of column space + Dimension of null sapce = $$n$$ = number of columns



Proof: 

$$A \in \R^{m \times n}$$ 

$$A \implies R_r$$ (row reduced Echelon)

$$Ax=0$$ and $$R_rx=0$$

Number of pivots in $$R_r=$$ columne rank ($$A$$)

$$R_rx=0$$ and $$Ax=0$$ 

Dim of null sapce for $$R_r=n-r$$  where $$n$$ is the number of columns and $$r$$ is the number of pivots

Because $$A$$ and $$R_r$$ are row equivalent, then

$$Ax=0$$ if and only if $$R_rx= \mathbf{0}$$

Dimension of null space of $$A=n-r$$

$$n-r+r=n$$

Dim($$n(A)$$)+Dim($$C(A)$$)=number of columns



Thm: 

Fundamental Theorem: $$A \in \R^{m \times n}$$

1. The row space of $$A$$ and nullsape of $$A$$ are orthogonal complements in $$\R^{n \times 1}$$

2. The column space of $$A$$ and left null sapce of $$A$$ are orthoginal complements in $$\R^{m \times 1}$$

Let $$v$$ be a vector space

$$u$$ be a subspace of $$v$$

$$w$$ be a subsapce of $$v$$

$$u$$ and $$w$$ are orthogonal complements means that $$\forall \alpha \in u$$ and $$\forall \beta \in w$$, $$\alpha \perp \beta$$

$$\alpha \cdot \beta=0$$

Proof:

$$Ax=y$$ where $$A \in \R^{m \times n}, x \in \R^{n \times 1}, y \in \R^{m \times 1}$$

1. Row Space: $$C(A^T)=\{x\in \R^{n \times 1}: A^Ty=x, y\in \R^{m \times 1}\}$$

   Null Space: $$n(A)=\{x\in \R^{n \times 1}: Ax= \mathbf{0}\}$$

   Assume $$\alpha \in C(A^T)$$ and $$\beta=n(A)$$

   $$\alpha \cdot \beta = \alpha^{T} \beta = (A^Ty)^Tx$$ 

   $$=y^TAx$$ where $$Ax=0$$

   $$=0$$

2. Column Space: $$C(A)=\{y \in \R^{m \times 1}: Ax=y, x \in \R^{n \times 1}\}$$

   Left null space $$n(A^T)=\{y \in \R^{m \times 1}: A^Ty=0\}$$

   Assume $$\alpha \in C(A)$$ and $$\beta \in n(A^T)$$

   $$\alpha^{T}\beta=(Ax)^Ty$$

   $$=x^TA^Ty$$ where $$A^Ty=0$$

   $$=0$$

Summary:

$$A \in \R^{m \times n}$$

column rank =$$r$$

dimension of nullspace= $$n-r$$

row rank = $$r$$

dimension of left null space =$$m-r$$

------

$$Ax=b$$

$$m \equiv $$ number of equations

$$n \equiv$$ number of unknowns

$$M=[Ab]$$

Rank($$M$$)          Rank($$A$$)

### Existence and Uniqueness

Thm:

Let $$Ax=b$$ be a system with $$n$$-unknowns $$m$$ equations and augmented matrix $$M=[A b]$$

1. The system has at least one solution if and only if $$rank(M)=rank(A)$$

   $$M'=\begin{bmatrix}1&2&3&4\\0&3&1&2\\0&0&2&1\\0&0&0&0\end{bmatrix}$$ $$A'=\begin{bmatrix}1&2&3\\0&3&1\\0&0&2\\0&0&0\end{bmatrix}$$

   $$n=3$$

2. The system has a unique solution if and only if $$rank(M)=n=rank(A)$$

### Inner Product Space

Vector space $$V$$ over a field $$F$$

Real Inner Product space:

Let $$V$$ be a vector space over field $$\R$$

$$<\alpha,\beta>$$ assisgn a real number for $$\alpha,\beta \in V$$

Then $$<\alpha,\beta>$$ is an inner product if:

[$$I_1$$] Linearity: $$<\alpha, a \beta+b \gamma>=a<\alpha, \beta>+b<\alpha, \gamma>$$ , $$\forall \alpha, \beta, \gamma \in V$$ and $$a,b \in \R$$

[$$I_2$$] Symmetry: $$<\alpha,\beta>=<\beta, \alpha>$$ , $$\forall \alpha, \beta \in V$$

[$$I_3$$] Positive Definite: $$<\alpha, \alpha> \geq 0$$ and $$<\alpha, \alpha>=0$$ if and only if $$\alpha=\mathbf{0}$$





Examples:

1. Euclidean $$\R^n$$ 

   $$<u,v>=u \cdot v=\sum_{i=1}^{n} u_i v_i$$

2. Function space $$c[a,b]$$ and polynomial space $$P_n(t)$$

   $$c[a,b]$$ - vector space of all continuous functions on the closed interval $$[a,b]$$

   $$<f,g>=\int_{a}^{b}f(x) g(x) dx$$

3. Matrix space $$M=\R^{m \times n}$$

   $$M$$ - vector space of all real $$m \times n$$ matrices

   $$<A,B>=Tr(B^T A)$$

------

## Week 4 Session 2

### Outlines

Orthogonality and Inner Products

Gram-Schmidt Process

------

Inner Product

$$<\alpha,\beta>$$

### Complex Inner Product Space

Vectorr $$V$$ over field $$\C$$

$$<\alpha,\beta>=\sum_{i=1}^{n}a_ib_i^{\ast}$$ where $$\alpha=\begin{bmatrix}a_1\\...\\a_n\end{bmatrix}$$ , $$\beta=\begin{bmatrix}b_1\\...\\b_n\end{bmatrix}$$

$$<\alpha,\beta>$$ must satisfy the following properties:

$$\forall \alpha, \beta, \gamma \in V ; \forall a, b \in \C$$ 

 [$$I_1$$] : Linearity

$$<\alpha, a\beta+b\gamma>=a^{\ast}<\alpha,\beta>+b^{\ast}<\alpha, \gamma>$$

[$$I_2$$] : Conjugate Symmetry

$$<\alpha, \beta>=<\beta, \alpha>^{\ast}$$

[$$I_3$$] : Positive Definite:

$$<\alpha, \alpha> \geq 0$$ and $$<\alpha, \alpha>=0 $$ if and only if $$\alpha=\mathbf{0}$$

------

### Normed Vector Spaces

Let $$V=\{\alpha, \beta, \gamma, ...\}$$ be a vector space over a field $$F$$. A norm $$||\cdot||$$ of $$V$$ is a function from the elements of $$v$$ (vectors in $$V$$) into the non-negative real number such that: 

[$$N_1$$] : $$||\alpha|| \geq 0$$ , $$\forall \alpha \in V$$ and $$||\alpha||=0$$ if an only if $$\alpha=\mathbf{0}$$

[$$N_2$$] : $$||k\alpha||=|k|||\alpha||$$, $$\forall \alpha \in V$$ and $$\forall k \in F$$

[$$N_3$$] : $$||\alpha+\beta|| \leq ||\alpha||+||\beta||$$, $$ \forall \alpha, \beta \in V$$ (triangle inequality) 



Example:

1. $$v=\R^n, \alpha \in V, \alpha=[a_1, ... a_n]$$

   $$||\alpha||=\sqrt{(a_1)^2+...(a_n)^2}$$   - Euclidean Norm

2. $$v = \C^n$$ Complex field


------



## Week 4 (Messed)

### Outlines

- Review

- Dimension

  Rank Theorem

  Counting Theorem

  Fundamental Theorem

  Existence and Uniqueness  

- Inner Product, Orthogonality

------

Review: 

Theorem 1:

Interchanging rows of a matrix leaves its row rank unchanged.



Theorem 2:

If $$Ax=0$$ and $$Bx=0$$ have the same solution then $$A$$ and $$B$$ have the same column rank

[Proof: Lecture 6]



Theorem 3: 

Elementary row operation does not change the column rank.

Reason: Elementary row operation preserves solution, then apply Theorem 2

------

$$Ax=b$$ where $$A\in \R^{m \times n}, x \in \R^{n \times 1}, b \in \R^{n \times 1}$$

### Theorem 4: Rank Theorem: 

Dimension of column space = Dimension of row space



Proof:

column rank = Dimension of column space

row rank = Dimension of row space

Let $$r=$$ row rank of $$A$$

$$c=$$ column rank of $$A$$



Claim 1: $$c \leq r$$

Proof:

$$A= \begin{bmatrix} ... & a_1 & ... \\ ... & ... & ... \\ ... & a_r & ... \\ ... & a_{r+1} &...\\ ... & ... & ... \\ ...& a_m & ...\end{bmatrix}$$, where  $$\begin{bmatrix} ... & a_1 & ... \\ ... & ... & ... \\ ... & a_r & ... \end{bmatrix}$$ is linearly independent rows, $$\begin{bmatrix}... & a_{r+1} &...\\ ... & ... & ... \\ ...& a_m & ...\end{bmatrix}$$ is linearly dependent rows

Let $$B=\begin{bmatrix} ... & a_1 & ... \\ ... & ... & ... \\ ... & a_r & ... \end{bmatrix}$$, where $$B \in \R^{r \times n}$$, $$D=\begin{bmatrix}... & a_{r+1} &...\\ ... & ... & ... \\ ...& a_m & ...\end{bmatrix}$$, where $$D \in \R^{(m-r) \times n}$$ 

Note: \$$\forall j: r+1 \leq j \leq m, \exists t_{ji}$$' s such that 

$$a_j=\sum_{i=1}^{r}t_{ji}a_i$$ - Linearly dependent rows

Let $$T=[t_{ji}]$$

$$D=TB$$

$$A=\begin{bmatrix} B\\D\end{bmatrix}=\begin{bmatrix} B\\TB\end{bmatrix}$$

So, $$Ax=\mathbf{0}$$ means $$\begin{bmatrix} B\\TB\end{bmatrix}x=\begin{bmatrix} Bx\\TBx\end{bmatrix}=\mathbf{0}$$

$$Ax=0$$ if and only if $$Bx=0$$

The column rank of $$A=c$$

so the column rank of $$B=c$$

Remember:

$$Bx=d\in \R^{r \times 1}$$ 

so the column space of $$B \subset R^{r \times 1}$$

$$Dim(C(B)) \leq D(\R^{r \times 1})$$, where $$C(B)$$ is ye column space

$$c \leq r$$



Claim 2: $$r \leq c$$ 

Proof: 

Definition of transpose

$$c=$$ row rank of $$A^T$$

$$r=$$ column rank of $$A^T$$ 

$$A= \begin{bmatrix} ... & \alpha_1 & ... \\ ... & ... & ... \\ ... & \alpha_c & ... \\ ... & \alpha_{c+1} &...\\ ... & ... & ... \\ ...& \alpha_n & ...\end{bmatrix}$$, where  $$\begin{bmatrix} ... & \alpha_1 & ... \\ ... & ... & ... \\ ... & \alpha_c & ... \end{bmatrix}$$ is linearly independent rows, $$\begin{bmatrix}... & \alpha_{c+1} &...\\ ... & ... & ... \\ ...& a_n & ...\end{bmatrix}$$ is linearly dependent rows

Let $$E=\begin{bmatrix} ... & \alpha_1 & ... \\ ... & ... & ... \\ ... & \alpha_c & ... \end{bmatrix}$$, where $$B \in \R^{c \times m}$$, $$F=\begin{bmatrix}... & \alpha_{r+1} &...\\ ... & ... & ... \\ ...& \alpha_n & ...\end{bmatrix}$$, where $$D \in \R^{(n-c) \times m}$$ 

Note: \$$\forall i: c+1 \leq i \leq n, \exists r_{ij}$$' s such that 

$$\alpha_i=\sum_{j=1}^{c}r_{ij}\alpha_j$$

Let $$R=[r_{ij}]$$

then $$F=RE$$

$$A^T=\begin{bmatrix}E\\F\end{bmatrix}=\begin{bmatrix}E\\RE\end{bmatrix}$$ 

$$A^Ty=\mathbf{0}$$ means $$\begin{bmatrix}E\\RE\end{bmatrix}y=\begin{bmatrix}Ey\\REy\end{bmatrix}=\mathbf{0}$$

$$A^Ty=\mathbf{0}$$ if and only if $$Ey=0$$

The column rank of $$A^T=r$$

So the column rank of $$E=r$$

Remember:

$$Ex=f \in \R^{c \times 1}$$

So the column space of $$E \subset \R^{c \times 1}$$

$$Dim(C(E)) \leq Dim(\R^{c \times 1})$$ , where $$C(E)$$ is the column space of $$E$$

Then $$ r \leq c$$



Therefore $$c=r$$

------

### Theorem 5: Counting Theorem

Dimension of column space + Dimension of null space =number of columns



Reason: Let $$R_r$$ be the row-reduced echelon form of $$A$$

- Row space of $$A$$=row space of $$R_r$$

  Because rows of $$R_r$$ are linear combinations of rows of $$A$$ and vice versa

- Column Space of $$A=$$ column space of $$R_r$$

  Because same solution for $$Ax=0$$ and $$R_rx=0$$

- Null space of $$A=$$ null space of $$R_r$$

  Because elementary row operation preserves solution

From $$R_r$$

$$n$$ is the number of variables

$$r$$ is the number of pivot variables

$$n-r$$ is the number of free variables

$$n=r+(n-r)$$

number of columns=$$Dim(C(R_r))+Dim(n(R_r))$$

$$=Dim(C(A))+Dim(n(A))$$

------

Similarly,

Dimension of row space + Dimension of left null space = Dimension of rows

------

### Theorem 6: Fundamental Theorem

1. The row space and null space of $$A$$ are orthogonal complements in $$\R^n$$
2. The column space and left null space of $$A$$ are orthogonal complements in $$\R^m$$

Proof:

Definition: Let $$V$$ be a vector space. Let $$U$$ be a subspace of $$V$$ and $$W$$ be a subspace of $$V$$. $$u$$ and $$w$$ are orthogonal complements in $$V$$ means that $$\forall u \in U$$ and $$\forall w \in W$$, $$u \perp w$$ ($$u \cdot w=0$$)

1. Row space: $$C(A^T)=\{A^Ty: y\in \R^{m \times 1}\}$$

   Null space: $$n(A)=\{x\in \R^{n \times 1}: Ax=\mathbf{0}\}$$

   Let $$A^Ty_1 \in C(A^T)$$ and $$x_0 \in n(A)$$

   $$(x_0)^T A^T y_1=((x_0)^T A^T y_1)^T$$ where it is a representation of transpose of scalar

   $$=y_1Ax_0$$ where $$Ax_0=\mathbf{0}$$

   $$=\mathbf{0}$$

2. Column space: $$C(A)=\{Ax: x\in \R^{n \times 1}\}$$

   Left null space: $$n(A^T)=\{z \in \R^{m \times 1}: A^Tz= \mathbf{0}\}$$

   Let $$Ax_1 \in C(A)$$ and $$z_0 \in n(A^T)$$

   $$(z_0)^TAx_1=((z_0)^T A x_1)^T$$ 

   $$=(x_1)^TA^Tz_0$$ where $$A^Tz_0=\mathbf{0}$$

   $$=\mathbf{0}$$

   ------

### Summary 

for $$A \in \R^{m \times n}$$

Column rank = Row rank = Rank = $$r$$

Dimension of the null space = $$n-r$$

Dimension of the left null space = $$m-r$$

------

### Theorem 1: Existence and Uniqueness

Let $$Ax=b$$ be a system with $$n$$ unknowns with augmented matrix $$M=[A|b]$$ then:

#### Existence

The system has at least one solution if and only if 

rank($$A$$)=rank($$M$$)

#### Uniqueness

The system has a unique solution if and only if 

rank($$A$$)=rank($$M$$)=$$n$$



Proof:

1. $$A$$ has no solution if and only if there exist a degenerate row $$[0,0,...,0|b]$$ in the echelon form of $$M$$
2. rank($$A$$)=$$n$$ if and only if no free variable

------

### Inner product and orthogonality

#### Real inner product space:

Let $$V$$ be a vector space over $$\R$$. Suppose that $$\forall \alpha, \beta \in V$$ $$<\alpha, \beta>$$ assigns a real number. Then $$<\alpha, \beta>$$ is an inner product on $$V$$ if

[$$I_1$$] Linearity: $$<\alpha, a \beta+b \gamma>=a<\alpha, \beta>+b<\alpha, \gamma>$$ , $$\forall \alpha, \beta, \gamma \in V$$ and $$a,b \in \R$$

[$$I_2$$] Symmetry: $$<\alpha,\beta>=<\beta, \alpha>$$ , $$\forall \alpha, \beta \in V$$

[$$I_3$$] Positive Definite: $$<\alpha, \alpha> \geq 0$$ and $$<\alpha, \alpha>=0$$ if and only if $$\alpha=\mathbf{0}$$



Examples:

1. Euclidean $$\R^n$$ 

   $$<u,v>=u \cdot v=\sum_{i=1}^{n} u_i v_i$$

2. Function space $$c[a,b]$$ and polynomial space $$P_n(t)$$

   $$c[a,b]$$ - vector space of all continuous functions on the closed interval $$[a,b]$$

   $$<f,g>=\int_{a}^{b}f(x) g(x) dx$$

3. Matrix space $$M=\R^{m \times n}$$

   $$M$$ - vector space of all real $$m \times n$$ matrices

   $$<A,B>=Tr(B^T A)$$

------

### Complex Inner product Space

Vector space $$V$$: $$\alpha, \beta, \gamma \in V$$

The field is $$\C$$: $$a,b \in \C$$

$$<u,v>$$ must satisfy the following:

[$$I_1$$] : Linearity

$$<\alpha, a\beta+b\gamma>=a^{\ast}<\alpha,\beta>+b^{\ast}<\alpha, \gamma>$$

[$$I_2$$] : Conjugate Symmetry

$$<\alpha, \beta>=<\beta, \alpha>^{\ast}$$

[$$I_3$$] : Positive Definite:

$$<\alpha, \alpha> \geq 0$$ and $$<\alpha, \alpha>=0 $$ if and only if $$\alpha=\mathbf{0}$$

------

### Normed Vector Spaces

Let $$V=\{\alpha, \beta, \gamma, ...\}$$ be a vector space over a field $$F$$. A norm $$||\cdot||$$ of $$V$$ is a function from the elements of $$v$$ (vectors in $$V$$) into the non-negative real number such that: 

[$$N_1$$] : $$||\alpha|| \geq 0$$ , $$\forall \alpha \in V$$ and $$||\alpha||=0$$ if an only if $$\alpha=\mathbf{0}$$

[$$N_2$$] : $$||k\alpha||=|k|||\alpha||$$, $$\forall \alpha \in V$$ and $$\forall k \in F$$

[$$N_3$$] : $$||\alpha+\beta|| \leq ||\alpha||+||\beta||$$, $$ \forall \alpha, \beta \in V$$ (triangle inequality) 



Example:

1. $$v=\R^n, \alpha \in V, \alpha=[a_1, ... a_n]$$

   $$||\alpha||=\sqrt{(a_1)^2+...(a_n)^2}$$   - Euclidean Norm

2. $$v = \C^n$$ Complex field

   $$\forall \alpha \in V$$, $$||\alpha||=\sqrt{(a_1)^2+...(a_n)^2}$$

------

Definition: A metric $$M(\alpha, \beta)$$ on pairs of elements $$\alpha, \beta \in V$$ satisfies the following:

[$$M_1$$] : $$M(\alpha, \beta)=0$$ if and only if $$\alpha=\beta$$

[$$M_2$$] : $$M(\alpha, \beta)=M(\beta, \alpha)$$

[$$M_3$$] : $$M(\alpha, \beta)+M(\beta, \gamma) \geq M(\alpha, \gamma), \forall \alpha, \beta, \gamma \in V$$

------

$$l^p$$ - distance

$$l^p(x,y)= \sqrt[p]{\sum_{i=1}^{n}|x_i-y_i|^p} , 1 \leq p \leq \infty$$

$$P=1$$

$$l^1(x,y)=\sum_{i=1}^{n}|x_i-y_i|$$ - Absolute

Let $$x,y \in B^n = \{0,1\}^n$$

consider $$x=\begin{bmatrix} 1 \\ 0 \\ 0 \\ 1 \\ 1 \end{bmatrix}$$ , $$y=\begin{bmatrix}1\\0\\1\\0\\1\end{bmatrix}$$ 















































































































































