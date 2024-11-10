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

Let $$(n<m)$$         - (Without Loss Of Generality)

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

Dimension Theorem

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

The column rank of $$A^T$$ is $$r$$

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

Number of pivots in $$R_r=$$ column rank ($$A$$)

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

dimension of null space= $$n-r$$

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

$$<\alpha,\beta>$$ assign a real number for $$\alpha,\beta \in V$$

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

Vector $$V$$ over field $$\C$$

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

### Metric Space

Vector space $$V$$ over $$F$$

$$M(\alpha, \beta)$$ - metric

Properties of a matric:

[$$M_1$$] : $$M(\alpha, \beta) \geq 0$$ and $$M(\alpha, \beta) =0$$ if and only if $$\alpha=\beta$$

[$$M_2$$] : $$M(\alpha, \beta)=M(\beta, \alpha)$$

[$$M_3$$] : $$M(\alpha, \gamma) \leq M(\alpha, \beta) + M(\beta, \gamma)$$

------

### Norm

$$l^p$$ - norm : $$\sqrt[p]{\sum_{i=1}^{n}|x_i|^p}=||x||_p$$

$$l^p$$ - distance : $$||x-y||_p$$

------

Volume of an Euclidean ball of radians $$\gamma$$

$$l^2$$ - norm $$r=\sqrt[2]{\sum_{i=1}^{n}x_i^2}=n=2$$

$$r=\sqrt{x^2+y^2}$$

Given Conditions:

$$V_n(r)=c_n r^n$$ ; $$c_n=\frac{2 \pi}{n}c_{n-2}$$

| $$n$$                                      | $$V_n$$                |
| ------------------------------------------ | ---------------------- |
| $$c_1=2$$                                  | $$2r$$                 |
| $$c_2=\pi$$                                | $$\pi r^2$$            |
| $$c_3=\frac{2 \pi}{3}c_1=\frac{4 \pi}{3}$$ | $$\frac{4 \pi}{3}r^3$$ |
| $$c_4=\frac{2 \pi}{4}c_2=\frac{\pi^2}{2}$$ | $$\frac{\pi^2}{2}r^4$$ |

![image-20240922181228013](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240922181228013.png)

$$ 0 < \epsilon <r$$

Volume shell - Entire Volume

$$\frac{c_nr^n-c_n(r-\epsilon)^n}{c_nr^n}$$

$$=\frac{r^n-(r-\epsilon)^n}{r^n}$$

$$=1-(1-\frac{\epsilon}{r})^n$$

$$0<\epsilon<r$$ $$\implies$$ $$0<\frac{\epsilon}{r}<1$$

$$1>1-\frac{\epsilon}{r}>0$$

$$lim_{n \rightarrow \infty}1-(1-\frac{\epsilon}{r})^n=1$$

------

### Orthogonality

Vector space $$V$$ over field $$F$$

$$\alpha, \beta \in V$$
$$\alpha \perp \beta$$ if and only if $$<\alpha,\beta>=0$$

Def: Let $$S=\{\alpha_1, ... \alpha_n\} \subset V$$ is mutually orthogonal if and only if

$$\alpha_i \cdot \alpha_j=0$$ for $$i \neq j$$

------

### Mutually Orthonormal 

A vector is normal if and only if its norm $$||\cdot||$$ is equal to 1

Def: Let $$S=\{\beta_1, ... \beta_n\} \subset V$$ is mutually orthonormal if and only if 

 $$ \alpha_i \cdot \alpha_j = \begin{cases} 0, \text{if\:} i \neq j \\ 1, \text{if\:} i =j \end{cases}$$

$$S=\{\alpha_1, ... \alpha_n\}$$ which is mutually orthogonal $$\implies$$ $$T=\{\frac{\alpha_1}{||\alpha_1||}, ...,\frac{\alpha_n}{||\alpha_n||} \}$$ which is mutually orthonormal

------

$$S$$ is linearly independent $$\centernot\implies$$ $$S$$ is mutually orthogonal

$$S$$ is mutually orthogonal $$\implies$$ $$S$$ is linearly independent



Example.

$$S=\{\alpha_1, \alpha_2, \alpha_3\}$$

where $$\alpha_1=\begin{bmatrix}1\\0\\0\end{bmatrix}$$ , $$\alpha_2=\begin{bmatrix}0\\1\\1\end{bmatrix}$$ , $$\alpha_3=\begin{bmatrix}0\\1\\-1\end{bmatrix}$$ 

$$v=\begin{bmatrix}3\\5\\2\end{bmatrix}$$

where $$\alpha_1 \cdot \alpha_2=0, \alpha_1 \cdot \alpha_3=0, \alpha_2 \cdot \alpha_3=0$$

$$c_1=\frac{v \cdot \alpha_1}{\alpha_1 \cdot \alpha_1}=\frac{3}{1}$$

$$c_2=\frac{v \cdot \alpha_2}{\alpha_2 \cdot \alpha_2}=\frac{7}{2}$$

$$c_3=\frac{v \cdot \alpha_3}{\alpha_3 \cdot \alpha_3}=\frac{3}{2}$$

Therefore: $$v=3\alpha_1+\frac{7}{2}\alpha_2+\frac{3}{2}\alpha_3$$



Thm: 

If $$S=\{\alpha_1, ... \alpha_n\}$$ is in a vector space $$V$$ and $$S$$ is mutually orthogonal (with $$\alpha_i \neq 0$$), then $$S$$ is linearly independent

Proof:

$$c_1\alpha_1+...c_n\alpha_n=\mathbf{0}$$

$$(c_1\alpha_1+...c_n\alpha_n)\cdot \alpha_i=\mathbf{0} \cdot \alpha_i=\mathbf{0}$$

$$\sum_{j=1}^{n}c_j(\alpha_j \cdot \alpha_i)=0$$

$$\sum_{j=1, j \neq i}^{n}c_j(\alpha_j \cdot \alpha_i)+c_i(\alpha_i \cdot \alpha_i)=0$$

$$\sum_{j=1, j \neq i}^{n}c_j(\alpha_j \cdot \alpha_i)=0$$ because $$S$$ is mutually orthogonal

$$c_i(\alpha_i \cdot \alpha_i)=0$$

$$c_i=\frac{0}{\alpha_i \cdot \alpha_i}$$ where $$\alpha_i \neq 0$$

Therefore, $$c_1=c_2=...=c_n$$ is the only solution

Therefore, $$S$$ is linearly independent



$$S=\{\alpha_1, \alpha_2, \alpha_3\}$$

where $$\alpha_1=\begin{bmatrix}1\\0\\0\end{bmatrix}$$ , $$\alpha_2=\begin{bmatrix}1\\1\\0\end{bmatrix}$$ , $$\alpha_3=\begin{bmatrix}1\\1\\1\end{bmatrix}$$ 

It is linearly independent but not mutually orthogonal 

where $$\alpha_1 \cdot \alpha_2=1, \alpha_1 \cdot \alpha_3=2, \alpha_2 \cdot \alpha_3=1$$



Thm:

If $$S=\{\alpha_1, ... \alpha_n\}$$ is in a vector space $$V$$ , $$S$$ is a basis of $$V$$ and $$S$$ is mutually orthogonal, then $$\forall \beta \in V, \exists a_i's$$ such that

$$a_1\alpha_1+...+a_n\alpha_n=\beta$$

$$a_i=\frac{\beta \cdot \alpha_i}{\alpha_i \cdot \alpha_i}$$

Proof:

$$S$$ is a basis for $$V$$

$$\forall \beta \in V$$

$$a_1\alpha_1+...+a_n\alpha_n=\beta$$

$$(a_1\alpha_1+...+a_n\alpha_n)\alpha_i=\beta \cdot \alpha_i$$

$$\sum_{j=1}^{n}a_j(\alpha_j \cdot \alpha_i)=\beta \alpha_i$$

$$\sum_{j=1, j \neq i}^{n}a_j(\alpha_j \cdot \alpha_i)+a_i(\alpha_i \cdot \alpha_i)=\beta \cdot \alpha_i$$

$$a_i(\alpha_i \cdot \alpha_i)=\beta \cdot \alpha_i$$

Therefore, $$a_i=\frac{\beta \cdot \alpha_i}{\alpha_i \cdot \alpha_i}$$

------

$$S=\{\alpha_1, ... \alpha_n\}$$ is a orthogonal basis of $$V$$

1. Basis
2. Mutually orthogonal

------

### Projection

Projection of $$\alpha$$ onto $$\beta$$

![image-20240922192152219](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240922192152219.png)

$$\gamma=Proj_{\beta}(\alpha)=c\beta$$

$$\omega=\alpha-\gamma$$ and $$\omega \perp \beta$$

$$(\alpha-\gamma) \perp \beta$$

$$(\alpha-c\beta) \perp \beta$$

$$(\alpha-c\beta) \cdot \beta=0$$

$$\alpha \cdot \beta - c \beta \cdot \beta=0$$

$$c=\frac{\alpha \cdot \beta}{\beta \cdot \beta}$$

$$Proj_{\beta}(\alpha)=c \beta=\frac{\alpha \cdot \beta}{\beta \cdot \beta}\beta$$

$$Orth_{\beta}(\alpha) \equiv \alpha - \gamma = \alpha - \frac{\alpha \cdot \beta}{\beta \cdot \beta}\beta$$

$$Proj_{\beta}(\alpha)+Orth_{\beta}(\alpha)=\alpha$$

------

$$w$$ is a subspace of $$V$$

Projection of $$\alpha$$ onto $$\omega$$ 

$$S=\{\beta_1, ... \beta_m\}$$ is an orthogonal basis for $$\omega$$

![image-20240922210630840](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240922210630840.png)

$$\gamma=Proj_{w}(\alpha)=c_1\beta_1+ ... + c_m\beta_m$$

1. $$\gamma \cdot \beta_i=(c_1\beta_1+ ... + c_m\beta_m) \cdot \beta_i$$

   $$=c_i(\beta_i \cdot \beta_i)+\sum_{j=1, j \neq i}^{m} c_j(\beta_j \cdot \beta_i)$$

2. $$\omega=\alpha-\gamma : \omega \perp \beta_i$$

   $$(\alpha-\gamma) \perp \beta_i$$

   $$(\alpha-\gamma) \cdot \beta_i=0$$

   $$\alpha \cdot \beta_i=\gamma \cdot \beta_i$$

Then $$\gamma \cdot \beta = \alpha \cdot \beta_i=c_i(\beta_i \cdot \beta_i)$$

$$c_i=\frac{\alpha \cdot \beta_i}{\beta_i \cdot \beta_i}$$

$$\gamma=\frac{\alpha \cdot \beta_1}{\beta_1 \cdot \beta_1}+...+\frac{\alpha \cdot \beta_m}{\beta_m \cdot \beta_m}=Proj_{\omega}(\alpha)$$

------

$$S=\{\alpha_1, ... \alpha_n\}$$ which is linearly independent "Gram Schmidt" $$\implies$$ $$T=\{\beta_1, ... \beta_n\}$$ which is mutually orthogonal 

$$L(S)=L(T)$$

Def:

If $$V$$ is a vector space and $$S$$ is a subspace of $$V$$, then

$$\omega=\{\alpha+\beta: \alpha \in S, \beta \in S^{\perp}\}=V=S \oplus S^{\perp}$$

where $$S^{\perp}$$ is the orthogonal complement of $$S$$

![image-20240922213439496](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240922213439496.png)

### Gram-Schmidt Process

Given $$S=\{\alpha_1, ... , \alpha_n\}$$ where $$S$$ is linearly independent.

Find $$T=\{\beta_1, ..., \beta_n\}$$ where $$S$$ is mutually orthogonal and $$L(s)=L(\tau)$$

$$\beta_1=\alpha_1$$

$$V_1=span\{\alpha_1\}=span\{\beta_1\}$$

$$\beta_2=\alpha_2-\frac{\alpha_2 \cdot \beta_1}{\beta_1 \cdot \beta_1}\beta_1$$ where $$Proj_{v_1}(\alpha_2)=\frac{\alpha_2 \cdot \beta_1}{\beta_1 \cdot \beta_1}\beta_1$$

$$V_2=span(\{\alpha_1, \alpha_2\})=span(\{\beta_1, \beta_2\})$$

$$\beta_3=\alpha_3-[\frac{\alpha_3 \cdot \beta_1}{\beta_1 \cdot \beta_1}\beta_1+\frac{\alpha_3 \cdot \beta_2}{\beta_2 \cdot \beta_2}\beta_2]$$ where $$Proj_{v_2}(\alpha_3)=\frac{\alpha_3 \cdot \beta_1}{\beta_1 \cdot \beta_1}\beta_1+\frac{\alpha_3 \cdot \beta_2}{\beta_2 \cdot \beta_2}\beta_1$$

$$\beta_k=\alpha_k-[\frac{\alpha_k \cdot \beta_1}{\beta_1 \cdot \beta_1}\beta_1+...+\frac{\alpha_k \cdot \beta_{(k-1)}}{\beta_{(k-1)} \cdot \beta_{(k-1)}}\beta_{(k-1)}]$$

![image-20240922214725824](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20240922214725824.png)

$$\beta_2=\alpha_2-\frac{\alpha_2 \cdot \beta_1}{\beta_1 \cdot \beta_1}\beta_1$$ 

------

Ex. $$\alpha_1=\begin{bmatrix}1\\1\\1\\1\end{bmatrix}$$ , $$\alpha_2=\begin{bmatrix}1\\2\\0\\1\end{bmatrix}$$ , $$\alpha_3=\begin{bmatrix}2\\1\\1\\0\end{bmatrix}$$ , $$\alpha_4=\begin{bmatrix}0\\0\\3\\1\end{bmatrix}$$ 

$$\beta_1=\alpha_1=\begin{bmatrix}1\\1\\1\\1\end{bmatrix}$$

$$\beta_2=\alpha_2-\frac{\alpha_2 \cdot \beta_1}{\beta_1 \cdot \beta_1}\beta_1=\begin{bmatrix}1\\2\\0\\1\end{bmatrix}-\frac{4}{4}\begin{bmatrix}1\\1\\1\\1\end{bmatrix}=\begin{bmatrix}0\\1\\-1\\0\end{bmatrix}$$

$$\beta_3=\alpha_3-[\frac{\alpha_3 \cdot \beta_1}{\beta_1 \cdot \beta_1}\beta_1+\frac{\alpha_3 \cdot \beta_2}{\beta_2 \cdot \beta_2}\beta_2]=\begin{bmatrix}2\\1\\1\\0\end{bmatrix}-\frac{4}{4}\begin{bmatrix}1\\1\\1\\1\end{bmatrix}-\frac{0}{2}\begin{bmatrix}0\\1\\-1\\0\end{bmatrix}=\begin{bmatrix}1\\0\\0\\-1\end{bmatrix}$$

$$\beta_4=\alpha_4-[\frac{\alpha_4 \cdot \beta_1}{\beta_1 \cdot \beta_1}\beta_1+\frac{\alpha_4 \cdot \beta_2}{\beta_2 \cdot \beta_2}\beta_2+\frac{\alpha_4 \cdot \beta_3}{\beta_3 \cdot \beta_3}\beta_3]=\begin{bmatrix}0\\0\\3\\1\end{bmatrix}-\frac{4}{4}\begin{bmatrix}1\\1\\1\\1\end{bmatrix}-\frac{-3}{2}\begin{bmatrix}0\\1\\-1\\0\end{bmatrix}-\frac{-1}{2}\begin{bmatrix}1\\0\\0\\-1\end{bmatrix}=\begin{bmatrix}-1/2\\1/2\\1/2\\-1/2\end{bmatrix}$$

------

Let $$V$$ be a vector where

$$V=\{\begin{bmatrix}a & 0\\0 &a\end{bmatrix}, \text{where\:} a \in \R\}$$

then $$\{\begin{bmatrix}1 & 0\\0 &1\end{bmatrix}\}$$ is a basis, and the dimension is 1



## Week 4 Session 1 (Messed)

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

------

## Mid Term 1 Review

### Sample 1

$$A=\begin{bmatrix}1&2&3\\3 &2&1\\2&1&3\end{bmatrix}$$

Find $$A^{-1}$$ if the inverse exists otherwise give sufficent reason.

Using Gaussian Jordan-Elimination

$$\begin{bmatrix} 1&2&3&1&0&0\\3&2&1&0&1&0\\2&1&3&0&0&1\end{bmatrix}$$



$$R_1 : R_1$$

$$R_2 : R_2-3R_1$$

$$R_3:R_3-2R_1$$

$$\begin{bmatrix} 1&2&3&1&0&0\\0&-4&-8&-3&1&0\\0&-3&-3&-2&0&1\end{bmatrix}$$



$$R_3:R_3-3/4 R_2$$

$$\begin{bmatrix} 1&2&3&1&0&0\\0&-4&-8&-3&1&0\\0&0&1&1/12&-1/4&1/3\end{bmatrix}$$



$$R_1:R_1-3R_3$$

$$R_2:R_2+8R_3$$

$$\begin{bmatrix} 1&2&0&3/4&3/4&-1\\0&-4&0&-7/3&-1&8/3\\0&0&1&1/12&-1/4&1/3\end{bmatrix}$$



$$R_2:-1/4R_2$$

$$\begin{bmatrix} 1&2&0&3/4&3/4&-1\\0&1&0&7/12&1/4&-2/3\\0&0&1&1/12&-1/4&1/3\end{bmatrix}$$



$$R_1:R_1-2R_2$$

$$\begin{bmatrix} 1&0&0&-5/12&1/4&1/3\\0&1&0&7/12&1/4&-2/3\\0&0&1&1/12&-1/4&1/3\end{bmatrix}$$



### Sample 2

Find the left null space of matrix $$A$$

$$A=\begin{bmatrix}1&2&3&4\\2&-2&-1&1\\-1&-8&-10&-11\end{bmatrix}$$

Find the dimension of $$C(A)$$ and $$C(A^T)$$



$$N(A^T)=\{y \in \R^{3 \times 1}, A^Ty=0\}$$

$$A^T=\begin{bmatrix}1&2 &-1\\2&-2&-8\\3&-1&-10\\4&1&-11\end{bmatrix}$$

$$R_2:R_2-2R_1$$

$$R_3:R_3-3R_1$$

$$R_4:R_4-4R_1$$

$$\begin{bmatrix}1&2 &-1\\0&-6&-6\\0&-7&-7\\0&-7&-7\end{bmatrix}$$

$$\begin{bmatrix}1&2 &-1\\0&-6&-6\\0&0&0\\0&0&0\end{bmatrix}$$

$$-6y_2-6y_3=0$$

$$y_2=-y_3$$

$$y_1+2y_2-y_3=0$$

$$y_1=3y_3$$

$$y=\begin{bmatrix}y_1\\y_2\\y_3\end{bmatrix}=\begin{bmatrix}3y_3\\-y_3\\y_3\end{bmatrix}=y_3\begin{bmatrix}3\\-1\\1\end{bmatrix}$$

Basis $$(n(A^T))=\begin{bmatrix}3\\-1\\1\end{bmatrix}$$



Null space of $$A$$ ($$Ax=0, x \in \R^{4 \times 1}$$)

$$A=\begin{bmatrix}1&2&3&4\\2&-2&-1&1\\-1&-8&-10&-11\end{bmatrix}$$

$$R_2:R_2-2R_1$$

$$R_3:R_3+R_1$$

$$\begin{bmatrix}1&2&3&4\\0&-6&-7&-7\\0&-6&-7&-7\end{bmatrix}$$

$$R_3:R_3-R_2$$

$$\begin{bmatrix}1&2&3&4\\0&-6&-7&-7\\0&0&0&0\end{bmatrix}$$ $$\implies$$ dimension is 2

According to the counting theorem

$$D(C(A))+D(N(A))=$$ number of columns =4

$$D(N(A))=2$$

$$D(C(A^T))+D(N(A^T))=$$ number of rows =3

$$D(N(A^T))=2$$



### Sample 3

Use the Gram-Schmidt procedure to construct a set of orthonormal set of $$\{[-1,1,0],[-1,0,1],[0,1,1]\}$$ 

Express $$v=\begin{bmatrix}2&3&5\end{bmatrix}$$ as linear combination of such orthonormal vectors.

$$\alpha_1=\begin{bmatrix}-1\\1\\0\end{bmatrix}$$ , $$\alpha_2=\begin{bmatrix}-1\\0\\1\end{bmatrix}$$ , $$\alpha_3=\begin{bmatrix}0\\1\\1\end{bmatrix}$$

$$\beta_1=\alpha_1=\begin{bmatrix}-1\\1\\0\end{bmatrix}$$

$$||\beta_1||^2=2$$



$$\beta_2=\alpha_2-\frac{\alpha_2 \cdot \beta_1}{\beta_1 \cdot \beta_1}\beta_1$$

$$=\begin{bmatrix}-1\\0\\1\end{bmatrix}-\frac{1}{2}\begin{bmatrix}-1\\1\\0\end{bmatrix}$$

$$=\begin{bmatrix}-1/2\\-1/2\\1\end{bmatrix}$$

$$||\beta_2||^2=3/2$$



$$\beta_3=\alpha_3-[\frac{\alpha_3 \cdot \beta_1}{\beta_1 \cdot \beta_1}\beta_1+\frac{\alpha_3 \cdot \beta_2}{\beta_2 \cdot \beta_2}\beta_2]$$

$$=\begin{bmatrix}0\\1\\1\end{bmatrix}--\frac{1}{2}\begin{bmatrix}-1\\1\\0\end{bmatrix}-\frac{1/2}{3/2}\begin{bmatrix}-1/2\\-1/2\\1\end{bmatrix}$$

$$=\begin{bmatrix}2/3\\2/3\\2/3\end{bmatrix}$$

$$||\beta_3||^2=12/9$$



The orthonormal basis are:

$$\gamma_1=\frac{1}{\sqrt{2}}\begin{bmatrix}-1\\1\\0\end{bmatrix}$$

$$\gamma_2=\sqrt{2/3}\begin{bmatrix}-1/2\\-1/2\\1\end{bmatrix}=\sqrt{1/6}\begin{bmatrix}-1\\-1\\2\end{bmatrix}$$

$$\gamma_3=\sqrt{9/12}\begin{bmatrix}2/3\\2/3\\2/3\end{bmatrix}=\frac{1}{\sqrt{3}}\begin{bmatrix}1\\1\\1\end{bmatrix}$$

$$v=\begin{bmatrix}2&3&5\end{bmatrix}$$

$$v=(v \cdot \gamma_1)\gamma_1+(v \cdot \gamma_2)\gamma2+(v \cdot \gamma_3)\gamma_3$$

$$v \cdot \gamma_1=\frac{1}{\sqrt{2}}$$

$$v \cdot \gamma_2=\frac{5}{\sqrt{6}}$$

$$v \cdot \gamma_3=\frac{10}{\sqrt{3}}$$

$$v=\frac{1}{\sqrt{2}}\gamma_1+\frac{5}{\sqrt{6}}\gamma2+\frac{10}{\sqrt{3}}\gamma_3$$

$$=1/2 \begin{bmatrix}-1\\1\\0\end{bmatrix} + 5/6 \begin{bmatrix}-1\\-1\\2\end{bmatrix} +10/3 \begin{bmatrix}1\\1\\1\end{bmatrix}$$

------

## Week 5 Session 1 (Only 1)

### Outlines

Linear Transformation (mappings)

Groups

Symmetric Group

Determinants

------

$$A$$ is $$n \times n$$

### Linear mapping

$$Ax=b$$
Non empty sets $$A$$ and $$B$$

$$f:A \rightarrow B$$

$$f$$ assigns a unique to $$a\in A$$ in $$B$$ 

$$A \equiv \text{Domain of\:}f, B \equiv \text{Codomain}$$

$$A' \subset A, f(A')=\{f(a):a\in A'\}$$

$$B' \subset B, f^{-1}(B')=\{a\in A:f(a)=b,b\in B'\}$$



### Matrix Mapping 

Let $$A\in K^{m \times n}$$ (field $$K$$ )

$$F_A$$ is the transformation determined by $$A

$$F_A: K^m \rightarrow K^n$$

For $$\alpha \in K^m$$, $$F_A(\alpha)=A\alpha$$

Composition of Mappings:

$$f:A \rightarrow B$$ and $$g:B \rightarrow C$$ 

$$g \cdot f: A \rightarrow C$$ or $$g \circ f: A \rightarrow C$$

$$(g \circ f)(\alpha)=g(f(\alpha))$$              $$\alpha \in A$$

Let $$f: A \rightarrow B$$

1. $$f$$ is injective (one to one) if 

   $$f(\alpha)=f(\alpha')$$

   $$\implies \alpha = \alpha'$$

2. $$f$$ is surjective (onto) if

   $$\forall \beta \in B, \exist \alpha \in A: f(\alpha)=\beta$$

3. $$f$$ is bijective (one to one correspondence) means that 

   $$f$$ is injective and surjective

Identify mapping:
$$f:A \rightarrow A$$

$$\mathbb{1}_A: \mathbb{1}_A(\alpha)=\alpha$$

Inverse mapping:
$$f:A \rightarrow B$$ and $$g:B \rightarrow A$$, $$g=f^{-1}$$ if 

$$f \circ g =\mathbb{1}_B$$ and $$g \circ f=\mathbb{1}_A$$

------

### Linear Mapping

Let $$v$$ and $$u$$ are vector spaces over field $$K$$ and 

$$F:v \rightarrow u$$ then $$F$$ is a linear mapping if :

1. For any $$\alpha,\beta \in v$$, $$F(\alpha+\beta)=F(\alpha)+F(\beta)$$
2. For any $$k\in K, \alpha \in v$$, $$F(k\alpha)=kF(\alpha)$$

Note:

$$F:v \rightarrow u$$ is a linear mapping if

For any $$a,b\in k$$ and $$\alpha,\beta \in v$$

$$F(a\alpha+b\beta)=aF(\alpha)+bF(\beta)$$

------

#### Example

Let $$F: \R^3 \rightarrow \R^2$$ be a projection onto the $$xy$$ plane, where $$F(x,y,z)=(x,y)$$. Is $$F$$ a linear mapping?

Let $$\alpha=(x_1,y_1,z_1)$$, $$\beta=(x_2,y_2,z_2)$$

$$F(\alpha+\beta)=F(x_1+x_2, y_1+y_2,z_1+z_2)$$

$$=(x_1+x_2,y_1+y_2)$$

$$=(x_1,y_1)+(x_2,y_2)$$

=$$F(\alpha)+F(\beta)$$



$$F(k\alpha)=F(kx_1,ky_1,kz_1)$$

$$=(kx_1,ky_1)$$

$$=k(x_1,y_1)$$

$$=kF(\alpha)$$

------

#### Example

$$G:\R^2\rightarrow \R^2$$ 

$$G(x,y)=(x+1,y+2)$$. Is $$G$$ a linear mapping?

Let $$\alpha=(x_1,y_1), \beta=(x_2,y_2)$$

$$G(\alpha)=(x_1+1,y_1+2)$$

$$G(\beta)=(x_2+1,y_2+2)$$

$$G(\alpha+\beta)=G(x_1+x_2,y_1+y_2)$$

$$=(x_1+x_2+1,y_1+y_2+2)$$

$$G(\alpha+\beta)-G(\alpha)=(x_2,y_2) \neq G(\beta)$$

$$G$$ is not a linear mapping

------

#### Example

$$J: v \implies \R$$

$$J(f(t))=\int_0^1{f(t)dt}$$

$$J(af(t)+bg(t))=\int_0^1{af(t)+bg(t)dt}$$

$$=\int_0^1af(t)dt+\int_0^1{bg(t)dt}$$

$$=a\int_0^1f(t)dt+b\int_0^1g(t)dt$$

$$=aJ(f(t))+bJ(g(t))$$

------

Thm: Let $$v$$ and $$u$$ be vector space over field $$K$$ and $$S=\{\alpha_1,\alpha_2,...\alpha_n\}$$ be a basis of $$v$$, then there exists a linear mapping $$F: V \rightarrow U$$ such that any $$\beta_1, \beta_2, ... \beta_n \in U$$ is a unique representation with respect to $$F$$ such that $$F(\alpha_i)=\beta_i$$

$$F: V \rightarrow U$$ where $$ V \rightarrow S=\{\alpha_1,\alpha_2,...\alpha_n\}, U \rightarrow \{\beta_1, \beta_2, ... \beta_n\}$$

Proof: 

1. Define $$F$$
2. $$F$$ is a linear mapping
3. $$F$$ is unique

Claim 1:

$$\gamma \in V$$

$$\gamma=a_1\alpha_1+...a_n\alpha_n$$                - $$a_1$$ is unique

$$F(\gamma)=a_1\beta_1+...+a_1\beta_n$$

$$F(\alpha_1)=F(1\alpha_1+0\alpha_2+...+0\alpha_n)=1\beta_1$$

$$F(\alpha v)=1\beta_i$$



Claim 2: $$F$$ is a linear mapping

Let $$v,w \in V$$

$$v=a_1\alpha_1+...a_n\alpha_n$$ 

$$w=b_1\alpha_1+...b_n\alpha_n$$

$$F(v)=\sum_{j=1}^{n}a_j\beta_j$$

$$F(w)=\sum_{j=1}^{n}b_j\beta_j$$

$$F(v+w)=F((a_1+b_1)\alpha_1+...+(a_n+b_n)\alpha_n)$$

$$=\sum_{j=1}^{n}(a_j+b_j)\beta_j$$

$$=\sum_{j=1}^{n}\alpha_j\beta_j+\sum_{j=1}^{n}b_j\beta_j$$

$$=F(v)+F(w)$$

$$F(kv)=F(k(a_1\alpha_1+...a_n\alpha_n))$$

$$=F(ka_1\alpha_1+...ka_n\alpha_n)$$

$$=\sum_{j=1}^{n}ka_j\beta_j$$

$$=k\sum_{j=1}^{n}a_j\beta_j$$

$$=kF(v)$$

$$F$$ is a linear mapping



Claim 3:

$$G: v \rightarrow u$$ is a linear mapping and $$G(\alpha_i)=\beta_i$$

$$G(a_1\alpha_1+...a_n\alpha_n)=\sum_{j=1}^{n}G(a_j\alpha_j)$$          - $$G$$ is a linear mapping

$$=\sum_{j=1}^{n}G(a_j\alpha_j)$$
$$=\sum_{j=1}^{n}a_jG(\alpha_j)$$

$$=\sum_{j=1}^{n}a_j\beta_j$$

$$=F(v)$$

=$$F(a_1\alpha_1+...a_n\alpha_n)$$

------

### Isomorphism

Definition

Two vector space $$v$$ and $$s$$ over field $$K$$ are isomorphic if there exist $$F: v \rightarrow u$$ such that 

1. $$F$$ is bijective
2. $$F$$ is a linear mapping

#### Example

vector space $$v$$ and $$s=\{\alpha_1,...\alpha_n\}$$ is a basis of $$v$$

$$Proj_{s}\alpha$$ for all $$\alpha$$ in $$v$$ is an isomorphism between $$v$$ and $$K^n$$ 

------

### Kernel and image of a linear mapping

$$F: v \rightarrow u$$

Kernel of $$F: Ker(F)$$

$$Ker(F)=\{\alpha\in v: F(\alpha)=0\}$$

Image of $$F: Im(F)$$

$$Im(F)=\{\beta \in u: \exist\alpha \in v, F(\alpha)=\beta\}$$

Nullity of $$F: Dim(Ker(F))$$

Rank of $$F:Dim(Im(F))$$

$$Dim(v)=Dim(Ker(F))+Dim(Im(F))$$

Note: 

$$Ker(F)$$ is a subspace of $$v$$

$$Im(F)$$ is a subspace of $$u$$



Thm:
Suppose $$\{\alpha_1,...,\alpha_n\}$$ spans $$V$$ and $$F: v \rightarrow u$$ is a linear mapping, then $$F(\alpha_1),...F(\alpha_n)$$ spans the image of $$F$$ ($$Im(F)$$)

Proof:

If $$\gamma \in v$$ then $$\gamma=a_1\alpha_1+...+a_n\alpha_n$$             - $$a_i's$$

$$F(\gamma)=F(a_1\alpha_1+...+a_n\alpha_n)$$

$$=\sum_{j=1}^{n}F(a_j\alpha_j)$$

$$=\sum_{j=1}^{n}a_jF(\alpha_j)$$

$$=a_1F(\alpha_1)+a_2F(\alpha_2)+...+a_nF(\alpha_n)$$

------

### Singularity

$$F:v \rightarrow u$$

$$Ker(F)=\{\mathbf{0}\}$$

If $$\exists \alpha \in v: \alpha \neq \mathbf{0} \text{\:and\:} F(\alpha)=0, \text{then\:} F$$ is singular

------

$$A \in n \times n$$

$$Det(A)=\sum_{\sigma \in S_n}sgn(\sigma) a_1 \sigma^1 a_2 \sigma^2...a_n\sigma^n$$ where $$sgn(\sigma)$$ is the parity

### Group

A collection of objects and a binary operation such that:

1.  Closure: $$\forall a,b \in G, a \ast b \in G$$
2. Associativity: $$\forall a,b,c \in G, a \ast (b \ast c)=(a \ast b)\ast c$$
3. Identity: $$\exist e \in G: \forall a \in G, a \ast e=e \ast a =a$$
4. Inverse: $$\forall a \in G, \exist a' \in G: a \ast a'=a' \ast a=e$$

Semi-group operation $$\ast$$: A collection of objects and a binary such that it satisfies 1, 2, 3

Abelian Group (commutative group):

$$(G, \ast)$$ is an Abelian group if $$(G, \ast)$$ is a group and $$\forall a,b \in G, a \ast b=b \ast a $$

------

### Ring

A collection of items and two operations (usually addition $$+$$ and multiplication $$\times$$) such that

1. $$(R,+)$$ is a commutative group
2. $$(R,\times)$$ 
   1. $$\forall a ,b \in R, a \times b \in R$$
   2. $$\forall a,b,c \in R, a \times(b+c)=(a \times b) +(a \times c)$$
   3. $$\forall a,b,c \in R, (a+b)\times c=(a \times c)+(b \times c)$$
   4. $$\forall a,b,c \in R, a \times (b \times c)=(a \times b)\times c$$
   5. $$\exist \circ \in R: 0 \times a =0 \:  \forall a \in R$$

### Field

A field $$F$$ is a ring where $$F'=$$ all the elements in $$F$$ without the 0-element and $$(F', \times)$$ is commutative group 

------

Symmetric Group $$S_x$$ on $$x$$

Group $$G:$$ $$|G|$$ is the order of $$G$$ = number of elements in $$G$$

$$S_x=\{\sigma: x \rightarrow x \text{\:such that\:} \sigma \text{\:is bijective}\}$$

$$\sigma - \text{shuffles\:} x$$ 

$$\sigma$$ is permutation of $$x\in X$$

$$|S_n|=n!=n(n-1)...2 \cdot 1$$

$$\sigma, \tau$$

$$\sigma \circ \tau$$ or $$\tau \circ \sigma$$

$$\sigma \circ \tau \in S_x$$ ,  $$\tau \circ \sigma \in S_x$$

------

#### Example

Let $$X=\{1,2,3\}$$

$$|S_x|=3!=6$$

$$S_x=\{e,\sigma_1,\sigma_2,\tau_1,\tau_2,\tau_3\}$$

$$e=\begin{bmatrix}1 & 2 & 3 \\ \downarrow & \downarrow & \downarrow\\ 1&2&3 \end{bmatrix}$$

$$\sigma_1=\begin{bmatrix}1 & 2 & 3 \\ \downarrow & \downarrow & \downarrow\\ 2&3&1 \end{bmatrix}$$

$$\sigma_2=\begin{bmatrix}1 & 2 & 3 \\ \downarrow & \downarrow & \downarrow\\ 3&1&2 \end{bmatrix}$$

$$\tau_1=\begin{bmatrix}1 & 2 & 3 \\ \downarrow & \downarrow & \downarrow\\ 1&3&2 \end{bmatrix}$$

$$\tau_2=\begin{bmatrix}1 & 2 & 3 \\ \downarrow & \downarrow & \downarrow\\ 3&2&1 \end{bmatrix}$$

$$\tau_3=\begin{bmatrix}1 & 2 & 3 \\ \downarrow & \downarrow & \downarrow\\ 2&1&3 \end{bmatrix}$$

Is $$\alpha_1 \circ \tau_1 = \tau_1 \circ \alpha_1$$

$$\alpha_1 \circ \tau_1=\tau_1=\begin{bmatrix}1 & 2 & 3 \\\downarrow & \downarrow & \downarrow\\ 1&3&2 \\ \downarrow & \downarrow & \downarrow\\ 3&2&1 \end{bmatrix}=\tau_2$$

$$\tau_1 \circ \sigma_1 = \begin{bmatrix}1 & 2 & 3 \\ \downarrow & \downarrow & \downarrow\\ 2&3&1\\ \downarrow & \downarrow & \downarrow\\2&1&3 \end{bmatrix}=\tau_3$$

sign or parity $$sgn(\sigma)$$

$$x=\{1,2,3\}, \sigma \in S_x$$ 

$$\sigma=\{\sigma^1, \sigma^2, \sigma^3\}$$

Example: $$\sigma=\sigma^1, \sigma_1^1=2,\sigma_1^2=1, \sigma_1^3=3$$

Even and odd parity

$$\sigma_1=\{2, 1, 3\}$$

number of $$i$$ and $$k$$ : $$i<k$$ but $$\sigma^i > \sigma^k$$

$$\tau_2=\{3,2,1\}$$
$$sgn(\alpha)=\prod_{i<k}\frac{\sigma^k-\sigma^i}{k-i}$$

For $$S_3$$: $$sgn(\sigma)=\frac{\sigma^3-\sigma^2}{3-2}\frac{\sigma^2-\sigma^1}{2-1}\frac{\sigma^3-\sigma^1}{3-1}$$

$$=\frac{1-2}{3-2}\frac{2-3}{2-1}\frac{1-3}{3-1}$$

$$=-1$$

------

## Week 6 Session 1

### Outline

Determinants

Eigenvalues and Eigenvectors

------

$$f: \R^n \rightarrow \R^n$$

$$x_{t+1}=f(x_t)$$

Fixed point: $$f(\hat{x})=\hat{x}$$

$$A\hat{x}=\hat{x}$$

------

$$A \in n \times n$$

$$Det(A)=\sum_{\sigma \in S_n}sgn(\sigma)a_1\sigma^1...a_n\sigma^n$$

$$sgn(\sigma)=\prod_{i>k}\frac{\sigma^i-\sigma^k}{i-k}$$

$$sgn(\sigma)=\begin{cases}+1 & \text{if\:} \sigma \text{\:is even} \\ -1 & \text{if\:} \sigma \text{\:is odd} \end{cases}$$

$$sgn(\sigma)=(-1)^{N(\sigma)}$$

$$N(\sigma)\equiv$$ number of $$(i,k)$$ such that $$(i>k)$$ but $$\sigma^i < \sigma^k$$

$$\sigma=\begin{bmatrix}1 & 2 & 3 \\ \downarrow & \downarrow & \downarrow\\ 2&3&1 \end{bmatrix}$$

$$i=\begin{bmatrix} 3 \\ 3 \\2\end{bmatrix}$$, $$k=\begin{bmatrix} 1 \\ 2 \\1\end{bmatrix}$$

$$N(\sigma)=2$$

$$sgn(\sigma)=(-1)^2=1$$

------

Facts:

1. Let $$g(x_1,...x_n)=\prod_{i>k}(x_i-x_k)$$

2. Let $$\sigma(g)=\prod_{i>k}(x_{\sigma^i}-x_{\sigma^k})$$

   $$\sigma(g)=\begin{cases} +g & \text{if\:} \sigma \text{\:is even}\\ -g & \text{if\:} \sigma \text{\:is odd}\end{cases}$$

   $$\sigma(g)=sgn(\sigma)g$$

3. Let $$\sigma, \tau \in S_n$$ , $$sgn(\sigma \circ \tau)=sgn(\sigma)sgn(\tau)$$

   $$sgn(\sigma \circ \tau)g=(\sigma \circ \tau)g$$

   $$=\sigma(\tau(g))$$

   $$=sgn(\sigma)\tau(g)$$

   $$=sgn(\sigma)sgn(\tau)g$$

   $$sgn(\sigma \circ \tau)=sgn(\sigma)sgn(\tau)$$

4. $$sgn(\sigma)=sgn(\sigma^{-1})$$

   $$\sigma \circ \sigma^{-1}=\epsilon$$

   $$sgn(\sigma \circ \sigma^{-1})=sgn(\sigma)sgn(\sigma^{-1})=sgn(\epsilon)=1$$

   check the two cases of $$sgn(\sigma)\in \{+1,-1\}$$

5. Let $$\sigma=j_1j_2...j_n$$ for scalar $$a_{ij}$$ and $$a_{j1^1}a_{j2^2}...a_{jn^n}=a_{1k_1}a_{2k_2}...a_{nk_n}$$

   $$\sigma(k_i)=i$$

   Let us assume that $$\tau=k_1k_2...k_n$$

   $$\tau(j_i)=i$$

   $$\tau(j_i)=\tau(\sigma(i))=i$$

   $$(\tau \circ \sigma)i=i$$

   $$\tau=\sigma^{-1}$$

------

Thm: If $$\sigma^i$$ and $$\sigma^j$$ are interchanged in $$\sigma=(\sigma^1,...\sigma^n)$$ to given $$\hat{\sigma}$$, then $$sgn(\hat{\sigma})=-sgn(\sigma)$$

Proof: $$\prod_{i>k}\frac{\sigma^i-\sigma^j}{i-j}$$

------

#### Example

$$A=\begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22}\end{bmatrix}$$

$$Det(A)=\sum_{\sigma \in S_2}sgn(\sigma)a_{1\sigma^1}a_{2\sigma^2}$$

$$S_2=\{\epsilon, \sigma\}$$

$$\epsilon=\begin{bmatrix}1 & 2 \\ \downarrow & \downarrow \\ 1 & 2 \end{bmatrix}$$

$$\sigma=\begin{bmatrix}1 & 2 \\ \downarrow & \downarrow \\ 2 & 1 \end{bmatrix}$$

$$sgn(\epsilon)=+1$$

$$sgn(\sigma)=-1$$

$$Det(A)=(+1)a_{11}a_{22}+(-1)a_{12}a_{21}$$

$$=a_{11}a_{22}-a_{12}a_{21}$$

------

### Properties of Determinants

1. $$Det(A)=Det(A^T)$$

   $$A=[a_{ij}]$$

   $$A^T=B=[b_{ij}]$$ where $$b_{ij}=a_{ij}$$

   $$Det(A)=\sum_{\sigma \in S_n}sgn(\sigma)a_{a\sigma^1}...a_{n\sigma^n}$$

   $$Det(A^T)=\sum_{\sigma \in S_n}sgn(\sigma)b_{1\sigma^1}...b_{n\sigma^n}$$

   $$=\sum_{\sigma \in S_n}sgn(\sigma)a_{\sigma^11}...a_{\sigma^nn}$$                        - $$A^T=B$$

   $$=\sum_{\sigma \in S_n, \tau= \sigma^{-1}}sgn(\sigma)a_{1\tau^1}...a_{n\tau^n}$$              - Fact 5

   $$=\sum_{\tau \in S_n} sgn(\tau)a_{a\tau^1}...a_{n\tau^n}$$                        - Fact 4

   $$=Det(A)$$

2. If $$A$$ is a square matrix and two rows (or columns) are interchanged to form $$B$$, then

   $$Det(A)=-Det(B)$$

3. If $$A$$ is a square matrix with a zero row (or zero column), then

   $$Det(A)=0$$

   $$\sum_{\sigma\in S_n}sgn(\sigma)a_{1\sigma^1}...a_{n\sigma^n}$$

4. If $$A$$ has two identical rows (or two identical columns), then the determinant 

   $$Det(A)=0$$

   $$A=\begin{bmatrix}... & R_i & ... \\ ... & R_j &... \end{bmatrix}$$ $$R_i=R_j$$

   $$A'=\begin{bmatrix}... & R_i & ... \\ ... & R_j &... \end{bmatrix}$$

   $$Det(A)=-Det(A)$$

   $$Det(A)=-Det(A')=-Det(A)\implies Det(A)=0$$ 

5. If scaling a row (or a column) by $$k$$ transforms a square $$A$$ to $$B$$, then

   $$Det(B)=kDet(A)$$

   $$A=\begin{bmatrix}... & R_i & ...  \end{bmatrix}$$

   $$\sum_{\sigma \in S_n}sgn(\sigma)a_{1\sigma^1}...a_{n\sigma^n}$$

   $$B=\begin{bmatrix}... & kR_i & ...  \end{bmatrix}$$

   $$\sum_{\sigma \in S_n}sgn(\sigma)b_{1\sigma^1}...b_{n\sigma^n}$$

   $$=\sum_{\sigma \in S_n}sgn(\sigma)k(a_{1\sigma^1}...a_{n\sigma^n})$$

   $$Det(B)=kDet(A)$$

6. $$R_i: R_i + kR_j$$

   If adding a scalar multiple of a row (or a column) to another transforms a square matrix $$A$$ to $$B$$, then

   $$Det(B)=Det(A)$$

   $$Det(B)=\sum_{\sigma \in S_n}sgn(\sigma)a_{1\sigma^1}...a_{i\sigma^{i}}+ka_{j\sigma^{i}}...a_{n\sigma^n}$$

   $$=\sum_{\sigma \in S_n}sgn(n)a_{1\sigma^1}...a_{i\sigma^i}...a_{n\sigma^n}+k\sum_{\tau \in S_n}sgn(\tau)(a_{1\tau^1...a_{j\tau^{i}}}...a_{n\tau^{n}})$$

   where $$a_{1\tau^1...a_{j\tau^{i}}}...a_{n\tau^{n}}=0$$

   $$=Det(A)$$

7. If $$E$$ is an elementary matrix and $$A$$ is a square matrix, then

   $$Det(EA)=Det(E)Det(A)$$

8. $$Det(AB)=Det(A)Det(B)$$

9. If $$A$$ is a diagonal matrix

   $$A=\begin{bmatrix} a_{11} & 0 & ... & 0  \\ 0 & a_{22} & ... & 0 \\... & ... & ... & ...\\0 & ... & 0& a_{nn}\end{bmatrix}$$

   then $$Det(A)=\prod_{i=1}^{n}a_{ii}$$

10. If $$A$$ is a triangular matrix, then

    $$Det(A)=\prod_{i=1}^{n}a_{ii}$$

11. $$Det(A^{-1})=(Det(A))^{-1}$$ if $$Det(A) \neq 0$$

Thm:

The following statements are equivalent

1. $$A$$ is invertible       - $$M=[A | I] \sim [I| A^{-1}]$$

2. $$Ax=0$$ has only the zero solution
3. $$Det(A) \neq 0$$

$$A=E_nE_{n-1}...E_1I$$

$$Det(A)=Det(E_n)...Det(E_1)Det(I) \neq 0$$

$$Det(E_n)\neq 0$$ , $$Det(E_1)\neq 0$$, $$Det(I) =1 $$ 

------

### Block matrix

$$M=\begin{bmatrix} A_{11} & A_{12}\\ A_{21} & A_{22}\end{bmatrix}$$

$$A_{11}$$ is $$r \times r$$

$$A_{22}$$ is $$s \times s$$

$$M_1=\begin{bmatrix} I & 0\\A_{21}A_{11}^{-1} & I \end{bmatrix}$$, $$M_2=\begin{bmatrix} A_{11} & A_{12}\\0 & A_{22}-A_{21}A_{11}^{-1}A_{12} \end{bmatrix}$$

$$M=M_1M_2=\begin{bmatrix} A_{11} & A_{12}\\ A_{21} & A_{22}\end{bmatrix}$$ 

If $$B=\begin{bmatrix}B_{11} & 0 & ... & 0\\ B_{21} & B_{22} & ... & 0 \\ ...&...&...&...\\ B_{n1} & B_{n2} & ... & B_{nn}\end{bmatrix}$$

$$Det(B)=\prod_{i=1}^{n}Det(B_{ii})$$

$$Det(M)=Det(M_1)Det(M_2)$$

$$=1\times Det(A_{11})Det(A_{22}-A_{21}A_{11}^{-1}A_{12})$$

$$=Det(A_{11})Det(A_{22}-A_{21}A_{11}^{-1}A_{12})$$

------

### Determinants and volume

Let $$u_1, ... u_n \in \R^n$$

$$V^T=\begin{bmatrix}... & ... &... &... \\ v_1 & v_2 & ... & v_n \\ ... & ... & ... & ...\end{bmatrix}$$

$$V=\begin{bmatrix}... & v_1 & ... \\ ... & v_2 & ...\\...& ... & ...\\...& v_n & ...\end{bmatrix}$$

volume enclosed by $$v_1, ... v_n$$ is $$|Det(v)|$$

------

### Cofactors and minors

$$A \equiv [a_{ij}]$$

$$M \equiv $$ Delete row $$i$$ and column $$j$$ from $$A$$ 

$$M_{ij} \in (n-1) \times (n-1)$$

![image-20241005001805304](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20241005001805304.png)

say $$A=\begin{bmatrix}4 & 5 & 7 \\ -2 & 1 & 0 \\ 3 & 8 & 7 \end{bmatrix}$$

then $$M_{12}=\begin{bmatrix} -2 & 0 \\ 3 & 7\end{bmatrix}$$

Minor: $$m_{ij}=Det(M_{ij})$$

Cofactor: $$c_{ij}=(-1)^{i+j}m_{ij}$$

$$Det(A)=\sum_{j=1}^{n}a_{ij}c_{ij}=\sum_{i=1}^{n}a_{ij}c_{ij}$$         - Laplace Expansion

Adjoint: $$A=[a_{ij}]$$

$$\tilde{A}=[c_{ij}]$$

$$Adj(A)=\widetilde{A}^T$$

$$A^{-1}=\frac{Adj(A)}{Det(A)}$$

------

### Eigenvalues and Eigenvectors

Fixed point:

$$x_{t+1}=f(x_t)$$

Fixed point $$\hat{x}$$ is such that $$f(\hat{x})=\hat{x}$$

Example

$$y=f(x)=x^2$$

$$f: \R \rightarrow \R$$

$$x = x^2$$

$$\hat{x}\in\{0,1\}$$



$$y=f(x)=x^3$$

$$f: \R \rightarrow \R$$

$$\hat{x}\in\{-1,0,1\}$$



$$Df=\frac{df(x)}{dx}=f(x)$$

$$f(x)=e^x$$ is the eigen function

$$\frac{d}{dx}e^{cx}=ce^{cx}$$ where $$c$$ is the eigen value

------

## Week 6 Session 2

### Outline

Eigenvalues and Eigenvectors

Linearly Independent Eigenvectors (LIE)

------

### Eigenvectors generalizes fixed points

$$f: \R^n \rightarrow \R^n$$

$$\hat{x}$$ is a fixed point for $$f$$ is $$f(\hat{x})=\hat{x}$$

Example

$$y=f(x)=x^2 ; \hat{x} \in \{0,1\}$$

$$y=f(x)=x^3 ; \hat{x} \in \{-1,0,1\}$$



$$D:f \rightarrow g$$

$$Df=\frac{df(x)}{dx}$$

$$\hat{f}=e^x$$ because $$\frac{d\hat{f}}{dx}=e^x=\hat{f}$$

$$\frac{d}{dx}e^{cx}=ce^{cx}$$ where $$c$$ is the eigen value and $$e^{cx}$$ is the eigenfunction

------

Matrix

$$A \in \C^{n \times n}$$

$$A: \C^n \rightarrow \C^n$$

$$\hat{x}$$ is a fixed point for $$A$$ if $$A\hat{x}=\hat{x}$$ 

Definition:

Suppose $$x\neq 0$$ and $$\lambda \in \C$$ if $$Ax=\lambda x$$ for $$A\in C^{n \times n}$$ then

$$\lambda$$ is the eigenvalue and $$x$$ is the corresponding eigenvector

$$Ax=\lambda x$$

Example

$$A=\begin{bmatrix}2 & 1 \\ 3 & 4 \end{bmatrix}$$

$$x_1=\begin{bmatrix} 1\\ 1 \end{bmatrix}$$ , $$Ax_1=\begin{bmatrix}3 \\ 7 \end{bmatrix}$$

![image-20241005003536388](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20241005003536388.png)

$$x_2=\begin{bmatrix} 1\\ -1 \end{bmatrix}$$ , $$Ax_2=\begin{bmatrix}2 & 1 \\ 3 & 4 \end{bmatrix}\begin{bmatrix}1 \\ -1 \end{bmatrix}=\begin{bmatrix}1 \\ -1 \end{bmatrix}$$

------

Eigenvectors are not unique

$$Ax=\lambda x$$

$$A(cx)=\lambda(cx)$$         - $$c \in \C$$

$$x_2=\begin{bmatrix} 1\\ -1 \end{bmatrix}$$ , $$x_3=\begin{bmatrix} -2\\ 2 \end{bmatrix}$$

$$Ax_2=\begin{bmatrix}2 & 1 \\ 3 & 4 \end{bmatrix}\begin{bmatrix}-2 \\ 2 \end{bmatrix}=\begin{bmatrix}-2 \\ 2 \end{bmatrix}$$

------

Recall: The following statements are equivalent for a square matrix $$A$$

1. $$A$$ is invertible
2. $$Ax=0$$ has the zero-vector as the only solution
3. $$Det(A)\neq 0$$

Thm: $$\lambda \in \C$$ is an eigenvalue of $$A$$ if and only if $$Det(A-\lambda I)=0$$

Proof: $$Ax=\lambda x$$ and $$x\neq 0$$ (Definition of eigenvector and eigenvalue)

$$Ax=\lambda I x$$

$$Ax-\lambda I x= \mathbf{0}$$

$$(A-\lambda I) x= \mathbf{0}$$ and $$x \neq 0$$

$$Det(A- \lambda I)=0$$

------

### Characteristic Polynomial 

$$P_A(\lambda)=Det(A-\lambda I)$$ 

$$P_A(\lambda)=0$$

Finding eigenvalues and eigenvectors

1. Define the characteristic polynomial $$P_A(\lambda)$$
2. Solve $$P_A(\lambda)=0$$
3. For each $$\lambda$$, solve for $$x$$ in $$(A-\lambda I)x=0$$

Example

$$A=\begin{bmatrix}2 & 1 \\ 3 & 4 \end{bmatrix}$$

Find its eigenvalues and eigenvectors

Step 1: $$P_A(\lambda)$$

$$P_A(\lambda)=Det(A-\lambda I)$$

$$=\begin{bmatrix}2-\lambda & 1 \\ 3 & 4-\lambda \end{bmatrix}$$

$$=(2-\lambda)(4-\lambda)-3$$

$$=\lambda^2-6\lambda+5$$

Step 2: Solve $$P_A(\lambda)=0$$

$$\lambda_1=1, \lambda_2=5$$

Step 3: 

For $$\lambda=1$$

$$(A-\lambda I)x= \mathbf{0}$$

$$\begin{bmatrix}2-1 & 1 \\ 3 & 4-1 \end{bmatrix}x=\mathbf{0}$$

$$\begin{bmatrix}1 & 1 \\ 3 & 3 \end{bmatrix}x=\mathbf{0}$$

$$x=\begin{bmatrix} z \\ -z \end{bmatrix}$$ for $$z \in \C , z\neq 0$$

$$x=\begin{bmatrix} 1 \\ -1\end{bmatrix}$$ is the corresponding eigenvector for $$\lambda=1$$



For $$\lambda=5$$

$$(A-\lambda I)x= \mathbf{0}$$

$$\begin{bmatrix}2-5 & 1 \\ 3 & 4-5 \end{bmatrix}x=\mathbf{0}$$

$$\begin{bmatrix}-3 & 1 \\ 3 & -1 \end{bmatrix}x=\mathbf{0}$$

$$x=\begin{bmatrix} z \\ 3z \end{bmatrix}$$ for $$z \in \C , z\neq 0$$

$$x=\begin{bmatrix} 1 \\ 3\end{bmatrix}$$ is the corresponding eigenvector for $$\lambda=5$$

------

### Eigenspace

$$A=\begin{bmatrix}1 & 0 \\ 0 & 1 \end{bmatrix}$$

$$\lambda_1=1, \lambda_2=1$$

$$Ax=\lambda x$$

$$Ix=1x$$

$$P_A(\lambda)=Det(A-\lambda I)=\prod_{i=1}^{n}(\lambda_{i}-\lambda)$$

$$P_A(\lambda)=(\lambda_1-\lambda)(\lambda_2-\lambda)=(1-\lambda)(5-\lambda)=\lambda^2-6\lambda+5$$

$$Det(A-\lambda I)=\prod_{i=1}^{n}(\lambda_i-\lambda)$$

$$Det(A-\lambda I)=\prod_{i=1}^{n}(\lambda_i-\lambda)^{m_j}$$  where $$k \leq n$$

$$\sum_{j=1}^{k}m_j=n$$

Let $$A$$ be a diagonal matrix

$$A=\begin{bmatrix} a_{11} & 0 & ... & 0  \\ 0 & a_{22} & ... & 0 \\... & ... & ... & ...\\0 & ... & 0& a_{nn}\end{bmatrix}$$

$$A-\lambda I=\begin{bmatrix} a_{11}-\lambda & 0 & ... & 0  \\ 0 & a_{22}-\lambda & ... & 0 \\... & ... & ... & ...\\0 & ... & 0& a_{nn}-\lambda\end{bmatrix}$$

$$Det(A-\lambda I)=\prod_{i=1}^{n}(a_{ii}-\lambda)=P_A(\lambda)$$ where $$a_{ii}=\lambda_{i}$$

Let $$A$$ be a triangular matrix

$$A=\begin{bmatrix} a_{11} & 0 & ... & 0  \\ a_{21} & a_{22} & ... & 0 \\... & ... & ... & 0\\a_{n1} & a_{n2} & ...& a_{nn}\end{bmatrix}$$

$$Det(A-\lambda I)=\prod_{i=1}^{n}(a_{ii}-\lambda)$$ where $$a_{ii}=\lambda_{i}$$

Thm: $$Det(A)=\prod_{i=1}^{n}\lambda_{i}$$

Proof: $$Det(A-I \lambda)=\prod_{i=1}^{n}(\lambda_{i}-\lambda)$$ where $$\lambda=0$$

$$Det(A)=\prod_{i=1}^{n}\lambda_{i}$$

------

### LIE (Linearly independent eigenvectors)

Distinct Eigenvalues $$\implies$$ LIE (Linearly independent eigenvectors)

LIE (Linearly independent eigenvectors)$$\centernot\implies$$ Distinct Eigenvalues 

Thm: If $$A \in \C^{n \times n}$$ has $$n$$ distinct eigenvalues $$\lambda_1, \lambda_2, ... \lambda_{n}$$ Then $$A$$ has $$n$$ linearly independent eigenvectors $$\alpha_1, \alpha_2, ... \alpha_{n}$$

Proof: Mathematical Induction



Basis step: 

Show that $$A$$ with a distinct eigenvalue has linearly independent eigenvector set

$$Ax=\lambda x$$ where $$A,x \in {1 \times 1}$$

$$cx=\lambda x, x \neq 0$$

$$\{x\}$$ is linearly independent



Induction step:

Induction hypothesis: Assume that $$\lambda_1, \lambda_2, ... \lambda_{n}$$ are unique (distinct) and $$\alpha_1, \alpha_2, ... \alpha_{n}$$ are linearly independent

If $$\lambda_1, \lambda_2, ... \lambda_{n+1}$$ are distinct eigenvalues then $$\alpha_1, \alpha_2, ... \alpha_{n+1}$$ are linearly independent for $$A \alpha_{i}=\lambda_{i}\alpha_{i}$$

$$P \rightarrow Q$$

Proof: By contradiction $$(\sim Q \rightarrow \sim P)$$

$$\sim Q: \exist c_{j}'s$$ such that $$\alpha_{n+1}=\sum_{j=1}^{n}c_{j}\alpha_{j}$$ 

$$A\alpha_{n+1}=\lambda_{n+1}\alpha_{n+1}$$

$$A\alpha_{n+1}=A\sum_{j=1}^{n}c_{j}\alpha_{j}=\sum_{j=1}^{n}c_{j}A\alpha_{j}=\sum_{j=1}^{n}c_{j}\lambda_{j}\alpha_{j}$$                $$Eq. 3$$

$$A\alpha_{n+1}=\lambda_{n+1}\alpha_{n+1}=\lambda_{n+1}\sum_{j=1}^{n}c_{j}\alpha_{j}=\sum_{j=1}^{n}c_{j}\lambda_{n+1}\alpha_{j}$$          $$Eq. 4$$

$$Eq. 3 - Eq. 4$$

$$\sum_{j=1}^{n}c_{j}(\lambda_{j}-\lambda_{n+1})\alpha_j=0$$ where $$\alpha_{j} \neq 0$$

$$\exist j: \lambda_{j}-\lambda_{n+1}=0$$ 

$$\lambda_{j}=\lambda_{n+1}$$

Contradiction $$\sim P$$

------

Thm: Projection matrix $$p: p^2 =p $$ and $$p=p^{T}$$, then $$A_{i}=0$$ or $$A_{i}=1$$

Proof: $$px=\lambda x$$

$$ppx=p\lambda x=\lambda p x =\lambda \lambda x=\lambda^2x$$

$$ppx=px=\lambda x$$

$$\lambda^2x=\lambda x$$

$$\lambda \in \{0,1\}$$



Thm: $$A$$ and $$A^{T}$$ have the same eigenvalues

Proof: $$Det(A^{T}-\lambda I)=Det(A^{T}-\lambda I^{T})$$

$$=Det((A-\lambda I)^{T})$$

$$=Det(A-\lambda I)$$ where $$Det(A)=Det(A^{T})$$

------

$$\begin{bmatrix}2 & 1 \\ 0 & 0 \end{bmatrix}$$

$$\begin{bmatrix}2-\lambda & 1 \\ 0 & -\lambda \end{bmatrix}$$

Thm: $$Tr(A)=\sum_{i=1}^{n}a_{ii}=\sum_{i=1}^{n}\lambda_{i}$$

Proof: $$Det(A-\lambda I)=\prod_{i=1}^{n}(\lambda_i-\lambda)=P_A(\lambda)$$

$$(x+a_1)(x+a_2)...(x+a_n)=x^{n}+x^{n-1}(a_1+a_2+...+a_n)+x^{n-2}(a_1a_2+a_1a_3+...+a_{n-1}a_{n})+ ... + (a_1a_2...a_{n})$$

RHS: $$\prod_{i=1}^{n}(\lambda_i-\lambda)=(-\lambda)^{n}+(-\lambda)^{n-1}(\lambda_1+\lambda_2+...+\lambda_{n})+(-\lambda)^{n-2}(\lambda_1\lambda_2+...\lambda_{n-1}\lambda_{n})$$

LHS: $$Det(A-\lambda I)$$

$$C=A-\lambda I = \begin{bmatrix} a_{11}-\lambda & a_{12} & ... & a_{1n}  \\ a_{21} & a_{22}-\lambda & ... & a_{2n} \\... & ... & ... & ...\\a_{n1} & a_{n2} & ...& a_{nn}-\lambda\end{bmatrix}$$

$$Det(A- \lambda I)=\sum_{\sigma \in S_n} sgn(\sigma)c_1\sigma^1...c_{n}\sigma^{n}$$

If look at the term of $$(-\lambda)^{n-1}$$

$$(a_{11}-\lambda)(a_{22}-\lambda)...(a_{nn}-\lambda)$$

$$(-\lambda)^{n}+(-\lambda)^{n-1}(a_{11}+...+a_{nn})$$

$$\sum_{i=1}^{n}\lambda_{i}=\sum_{i=1}^{b}a_{ii}$$ 

coefficient of $$(-\lambda)^{n-1}$$

## Week 7 Session 1

### Outline

Similar matrices

Diagonalizable matrices

Power and exponential of matrices

Stability of differential equation

------

$$A=\begin{bmatrix} a & b \\ c & d\end{bmatrix}$$

$$T \equiv Tr(A)=a+d$$

$$D \equiv Det(A)=ad-bc$$

$$P_A(\lambda)=Det(A-\lambda I)=Det\begin{bmatrix} a-\lambda & b \\ c & d-\lambda\end{bmatrix}$$

$$=(a-\lambda)(d-\lambda)-bc$$

$$=\lambda^2-(a+d)\lambda+ad-bc$$
$$=\lambda^2-T\lambda+D$$

Let $$P_A(\lambda)=0$$

$$\lambda^2-T\lambda+D=0$$

$$\lambda=\frac{T \pm \sqrt{T^2-4D}}{2}$$



Thm: 

$$A$$ and $$A^T$$ have the same eigenvalues

Proof: 

$$P_{A^T}(\lambda)=Det(A^T-\lambda I)=Det(A^T-\lambda I^T)$$

$$=Det((A-\lambda I)^T)$$

$$=Det(A-\lambda I)$$

$$=P_A(\lambda)$$

------

### Similar matrices

$$A$$ is similar to $$B$$ ($$A \sim B$$)

Definition: $$A$$ is similar to $$B$$ is there exists an invertible matrix $$T$$ such that

$$AT=TB$$

$$T^{-1}AT=B$$

$$A=TBT^{-1}$$



Thm:

 If $$A \sim B$$, then $$A$$ and $$B$$ have the same eigenvalues

Proof:

Assume that $$A \sim B$$

$$B=T^{-1}AT$$

$$P_B(\lambda)=Det(B-\lambda I)=Det(T^{-1}AT-\lambda I)$$

$$=Det(T^{-1}AT-\lambda T^{-1}T)$$

$$=Det(T^{-1}(AT-\lambda T))$$

$$=Det(T^{-1}(A-\lambda I)T)$$

$$=Det(T^{-1})Det(A-\lambda I)Det(T)$$

$$=(Det(T))^{-1}Det(A-\lambda I)Det(T)$$

$$=Det(A-\lambda I)$$

$$=P_A(\lambda)$$



#### Example

$$A=\begin{bmatrix} 1 & 0 \\ 0 & 1\end{bmatrix}$$

$$\lambda_1=\lambda_2=1$$



$$B=\begin{bmatrix} 1 & 2 \\ 0 & 1\end{bmatrix}$$

$$\lambda_1=\lambda_2=1$$



$$T^{-1}AT=T^{-1}IT$$

$$=T^{-1}T=I\neq B$$

------

Let $$x'=Ax$$ be the initial coordinate system

$$T \gamma=AT\beta$$

$$\gamma=T^{-1}AT \beta$$ be the new coordinate system

$$T=\begin{bmatrix}... & ... & ... & ... \\ \alpha_1 & \alpha_2 & ... & \alpha_n\\ ... & ... & ... & ...\\\end{bmatrix}$$ where $$\alpha_1,\alpha_2,...\alpha_n$$ is basis of the new coordinate system

$$x'=Ax$$

$$\exist c_i's: x=c_1\alpha_1+c_2\alpha_2+...c_n\alpha_n$$

$$x=\begin{bmatrix}... & ... & ... & ... \\ \alpha_1 & \alpha_2 & ... & \alpha_n\\ ... & ... & ... & ...\\\end{bmatrix} \begin{bmatrix}c_1 \\ c_2 \\...\\ c_n \end{bmatrix}=T\beta$$ where $$T=\begin{bmatrix}... & ... & ... & ... \\ \alpha_1 & \alpha_2 & ... & \alpha_n\\ ... & ... & ... & ...\\\end{bmatrix}$$ $$\beta=\begin{bmatrix}c_1 \\ c_2 \\...\\ c_n \end{bmatrix}$$

$$x'=\begin{bmatrix}... & ... & ... & ... \\ \alpha_1 & \alpha_2 & ... & \alpha_n\\ ... & ... & ... & ...\\\end{bmatrix} \begin{bmatrix}d_1 \\ d_2 \\...\\ d_n \end{bmatrix}=T \gamma$$ where $$\gamma=\begin{bmatrix}d_1 \\ d_2 \\...\\ d_n \end{bmatrix}$$

Note:

$$T$$ is invertible because $$\alpha_1, ... \alpha_n$$ are linearly independent

$$\beta=T^{-1}x$$ , $$\gamma=T^{-1}x'$$

$$\gamma=T^{-1}x'$$              

$$\gamma=T^{-1}Ax$$            - $$x'=Ax$$

$$\gamma=T^{-1}AT \beta$$         - $$x=T \beta$$

Recall:

$$A$$ is similar to $$B$$ if $$\exist$$ invertible $$T$$ such that

$$AT=TB$$

$$A=TBT^{-1}$$

$$B=T^{-1}AT$$



$$x'=Ax$$ 

$$\gamma=T^{-1}AT\beta$$ where $$B=T^{-1}AT$$

$$T:$$ Transforms $$\beta$$ to the initial coordinate

$$A:$$ Linear transformation in the initial coordinate

$$T^{-1}:$$ Transforms back to the new coordinate

------

Note:

1. $$A:A \sim A$$

2. $$A \sim B \implies B \sim A$$

   $$B=T^{-1}AT$$

   Let $$S=T^{-1}, S^{-1}=T$$

   $$TBT^{-1}=TT^{-1}ATT^{-1}$$

   $$A=TBT^{-1}$$

   $$A=S^{-1}BS$$

3. If $$A \sim B$$ and $$B \sim C$$, then $$A \sim C$$

------

### Diagonalizable matrices

Definition: $$A$$ is diagonalizable if there exists a diagonal matrix $$\Lambda$$ such that $$A$$ is similar to $$\Lambda$$  

$$A \sim \Lambda$$

$$A=T\Lambda T^{-1}$$ or $$\Lambda=T^{-1}AT$$

Thm:

$$A$$ is diagonalizable if and only if $$A$$ has linearly independent eigenvectors (LIE)

$$P$$ if and only if $$Q$$ must satisfies:

Claim 1: $$P \rightarrow Q$$

Claim 2: $$Q \rightarrow P$$



Claim 1: $$A$$ is diagonalizable $$\implies A$$ has linearly independent eigenvectors

Proof: 

$$AT=T\Lambda$$                            - Assumption $$T$$ is invertible

$$A\begin{bmatrix}... & ... & ... & ... \\ \alpha_1 & \alpha_2 & ... & \alpha_n\\ ... & ... & ... & ...\\\end{bmatrix}=\begin{bmatrix}... & ... & ... & ... \\ \alpha_1 & \alpha_2 & ... & \alpha_n\\ ... & ... & ... & ...\\\end{bmatrix} \begin{bmatrix} \lambda_1 & 0 & ... & 0  \\ 0 & \lambda_2 & ... & 0 \\... & ... & ... & ...\\0 & ... & 0& \lambda_{n}\end{bmatrix}$$ where $$\Lambda=\begin{bmatrix} \lambda_1 & 0 & ... & 0  \\ 0 & \lambda_2 & ... & 0 \\... & ... & ... & ...\\0 & ... & 0& \lambda_{n}\end{bmatrix}$$

$$\begin{bmatrix}... & ... & ... & ... \\ A \alpha_1 & A \alpha_2 & ... & A \alpha_n\\ ... & ... & ... & ...\\\end{bmatrix}=\begin{bmatrix}... & ... & ... & ... \\ \alpha_1 \lambda_1 & \alpha_2 \lambda_2 & ... & \alpha_n \lambda_n\\ ... & ... & ... & ...\\\end{bmatrix}$$

$$A \alpha_i =\lambda_i \alpha_i$$ for $$i=\{1,2,...,n\}$$

$$\alpha_i$$ is an eigenvector

$$\alpha_1, ... \alpha_n$$ are linear independent                 - $$T$$ is invertible

$$A$$ has linearly independent eigenvectors



Claim 2: $$A$$ has linearly independent eigenvectors $$\implies A$$ is diagonalizable

Proof:

Let $$\alpha_1, ... \alpha_n$$ be the eigenvectors of $$A$$ and $$T=\begin{bmatrix}... & ... & ... & ... \\ \alpha_1 & \alpha_2 & ... & \alpha_n\\ ... & ... & ... & ...\\\end{bmatrix}$$

$$AT=A\begin{bmatrix}... & ... & ... & ... \\ \alpha_1 & \alpha_2 & ... & \alpha_n\\ ... & ... & ... & ...\\\end{bmatrix}$$

$$=\begin{bmatrix}... & ... & ... & ... \\ A\alpha_1 & A\alpha_2 & ... & A\alpha_n\\ ... & ... & ... & ...\\\end{bmatrix}$$

$$=\begin{bmatrix}... & ... & ... & ... \\ \lambda_1\alpha_1 & \lambda_1\alpha_2 & ... & \lambda_n\alpha_n\\ ... & ... & ... & ...\\\end{bmatrix}$$

$$=\begin{bmatrix}... & ... & ... & ... \\ \alpha_1 & \alpha_2 & ... & \alpha_n\\ ... & ... & ... & ...\\\end{bmatrix} \begin{bmatrix} \lambda_1 & 0 & ... & 0  \\ 0 & \lambda_2 & ... & 0 \\... & ... & ... & ...\\0 & ... & 0& \lambda_{n}\end{bmatrix}$$

$$=T \Lambda$$

$$T$$ is invertible 

$$\alpha_1, ... \alpha_n$$ are linear independent

$$A$$ is diagonalizable if and only if $$A$$ has linearly independent eigenvectors

------

#### Example

$$A=\begin{bmatrix} 2 & 1 \\ 3 & 4\end{bmatrix}$$

$$AE=E\Lambda$$

$$A=E \Lambda E^{-1}$$

$$\lambda=\frac{T \pm \sqrt{T^2-4D}}{2}$$

$$T=2+4=6$$

$$D=8-3=5$$

$$\lambda_1=5, \lambda_2= 1$$

For $$\lambda_1: $$

$$(A-\lambda I)x=\mathbf{0}$$

$$\begin{bmatrix} 2-5 & 1 \\ 3 & 4-5\end{bmatrix}x=\mathbf{0}$$

$$\begin{bmatrix} -3 & 1 \\ 3 & -1\end{bmatrix}x=\mathbf{0}$$

$$x=\begin{bmatrix} 1z \\ 3z \end{bmatrix}$$

$$\alpha_1=\begin{bmatrix} 1 \\ 3 \end{bmatrix}$$

For $$\lambda_2: $$

$$(A-\lambda I)x=\mathbf{0}$$

$$\begin{bmatrix} 2-1 & 1 \\ 3 & 4-1\end{bmatrix}x=\mathbf{0}$$

$$\begin{bmatrix} 1 & 1 \\ 3 & 3\end{bmatrix}x=\mathbf{0}$$

$$x=\begin{bmatrix} z \\ -z \end{bmatrix}$$

$$\alpha_2=\begin{bmatrix} 1 \\ -1 \end{bmatrix}$$

$$\Lambda=\begin{bmatrix} \lambda_1 & 0 \\ 0 & \lambda_2\end{bmatrix}=\begin{bmatrix} 5 & 0 \\ 0 & 1\end{bmatrix}$$

$$E=\begin{bmatrix} \alpha_1 & \alpha_2 \end {bmatrix}=\begin{bmatrix} 1 & 1 \\ 3 & -1\end{bmatrix}$$

$$A=E\Lambda E^{-1}$$

$$E^{-1}=\frac{1}{(1)(-1)-(1)(3)}\begin{bmatrix} -1 & -1 \\ -3 & 1\end{bmatrix}$$

Inverse of a matrix:

If $$A=\begin{bmatrix} a & b \\ c & d\end{bmatrix}$$

$$A^{-1}=\frac{1}{ad-bc}\begin{bmatrix} d & -b \\ -c & a\end{bmatrix}$$ where $$ad \neq bc$$

------

$$\frac{df}{dx}=f, f=e^{x}$$

$$\frac{de^{cx}}{dx}=ce^{cx}$$ 

------

Power of matrices

$$A^k=AAAAA...A$$ , $$k$$ in total

Assume that $$A$$ is diagonalizable

$$A^k=AAA...A$$

$$=E\Lambda E^{-1} E\Lambda E^{-1} ... E\Lambda E^{-1}$$   where $$E^{-1}E=I$$

$$=E\Lambda^{k} E^{-1}$$

$$=E\begin{bmatrix} \lambda_1^k & 0 & ... & 0  \\ 0 & \lambda_2^k & ... & 0 \\... & ... & ... & ...\\0 & ... & 0& \lambda_{n}^k\end{bmatrix}E^{-1}$$

Thm:

If $$A$$ has eigenvalues $$\lambda_1,\lambda_2,...\lambda_n$$ then $$A^k$$ has eigenvalues $$\lambda_1^k, \lambda_2^k, ... \lambda_n^k$$ for $$k=1,2,...$$

Proof:

Basos step

Induction step



Thm:

If $$A$$ is invertible and $$A$$ has eigenvalues $$\lambda_1,\lambda_2, ... \lambda_n$$ then $$A^{-1}$$ has eigenvalues $$\lambda_1^{-1}, \lambda_2^{-1},..., \lambda_n^{-1}$$

Proof: 

$$A\alpha=\lambda \alpha$$

$$A^{-1}A\alpha=A^{-1}\lambda \alpha$$

$$I \alpha=\lambda A^{-1} \alpha$$ 

$$A^{-1}\alpha=\frac{1}{\lambda}\alpha=\lambda^{-1}\alpha$$

------

$$e^{x}=\sum_{n=0}^{\infty}\frac{x^{n}}{n!}$$

$$e^{A}=\sum_{n=0}^{\infty}\frac{A^{n}}{n!}$$

Assume that $$A$$ is disgonalizable

$$A=E\Lambda E^{-1}$$

$$e^{A}=\sum_{n=0}^{\infty}\frac{(E\Lambda E^{-1})^{n}}{n!}$$

$$=\sum_{n=0}^{\infty}\frac{E\Lambda^{n} E^{-1}}{n!}$$

$$=E(\sum_{n=0}^{\infty}\frac{\Lambda^{n} }{n!} )E^{-1}$$

$$=E\begin{bmatrix} \sum_{n=0}^{\infty}\frac{\lambda_1^{n} }{n!} & 0 & ... & 0  \\ 0 & \sum_{n=0}^{\infty}\frac{\lambda_2^{n} }{n!} & ... & 0 \\... & ... & ... & ...\\0 & ... & 0& \sum_{n=0}^{\infty}\frac{\lambda_1^{n} }{n!}\end{bmatrix}E^{-1}$$

$$=E\begin{bmatrix} e^{\lambda_1} & 0 & ... & 0  \\ 0 & e^{\lambda_2} & ... & 0 \\... & ... & ... & ...\\0 & ... & 0& e^{\lambda_n}\end{bmatrix}E^{-1}$$

------

Thm:

If $$AB=BA$$, then $$e^{A+B}=e^Ae^B$$

Proof:

Binomial Theorem: $$(x+y)^n=\sum_{j=0}^{n} \binom{n}{j}x^{j}y^{n-j}$$

$$(x+y)^2=(x+y)(x+y)=x^2+xy+yx+y^2=x^2+2xy+y^2$$

$$e^Ae^B=(\sum_{i=0}^{\infty}\frac{A^i}{i!})(\sum_{j=0}^{\infty}\frac{B^j}{j!})$$

$$=\sum_{i=0}^{\infty}\sum_{j=0}^{\infty}\frac{A^iB^j}{i!j!}$$

Let $$l=i+j$$

$$=\sum_{l=0}^{\infty}\sum_{j=0}^{l}\frac{A^{l-j}B^j}{(l-j)!j!}$$

$$=\sum_{l=0}^{\infty} \frac{1}{l!}\sum_{j=0}^{l} \frac{l!}{j!(l-j)!}A^{l-j}B^j$$           - $$\binom{n}{j}=\frac{n!}{j!(n-j)!}$$

$$=\sum_{l=0}^{\infty} \frac{1}{l!} \binom{l}{j}A^{l-j}B^j$$                             

$$=\sum_{l=0}^{\infty} \frac{1}{l!} (A+B)^{l}$$                               - $$AB=BA$$

$$=e^{A+B}$$

------

#### Example

$$A= \begin{bmatrix} -2 & 1 \\ 1 & -2 \end{bmatrix}$$

$$\lambda_1=-1, \lambda=-3$$

$$T=-4, D=3$$

$$\alpha_1=\begin{bmatrix}1 \\ 1 \end{bmatrix}, \alpha_2=\begin{bmatrix} 1 \\-1 \end{bmatrix}$$

$$e^{At}=Ee^{\Lambda t}E^{-1}$$

$$E=\begin{bmatrix} 1 & 1 \\ 1 & -1\end{bmatrix}$$

$$E^{-1}=\frac{1}{-2} \begin{bmatrix}-1  & -1 \\ -1 & 1 \end{bmatrix}$$

$$e^{At}=\begin{bmatrix} 1 & 1 \\ 1 & -1\end{bmatrix} \begin{bmatrix} e^{-t} & 0 \\ 0 & e^{-3t}\end{bmatrix} \frac{1}{-2} \begin{bmatrix}-1  & -1 \\ -1 & 1 \end{bmatrix}$$

$$\lim_{t \rightarrow \infty}e^{At}=\begin{bmatrix} 1 & 1 \\ 1 & -1\end{bmatrix} \begin{bmatrix} \lim_{t \rightarrow \infty}e^{-t} & 0 \\ 0 & \lim_{t \rightarrow \infty}e^{-3t}\end{bmatrix} \frac{1}{-2} \begin{bmatrix}-1  & -1 \\ -1 & 1 \end{bmatrix}$$

$$\lim_{t \rightarrow \infty}e^{-t}=0$$

$$\lim_{t \rightarrow \infty}e^{-3t}=0$$

$$\frac{du}{dt}=u \implies u=e^{t}$$

$$\frac{du}{dt}=au \implies u=e^{at}$$

$$\frac{du}{dt}=Au \implies u=e^{At}$$

------

## Week 8 Session 1

### Content

Differential equation and matrix exponential

Stability of differential equation

Complex matrices

------

Review:

1. $$A$$ is similar to $$B$$ means there exists an invertible matrix $$T$$ such that $$AT=TB, (T^{-1}AT=B)$$

2. $$A$$ is diagonalizable means there exist invertible matrix $$T$$ and diagonal matrix $$\Lambda$$ such that $$AT=T\Lambda$$

3. $$A$$ is diagonalizable if and only if $$A$$ has linearly independent eigenvectors

4. $$e^{A}=\sum_{j=0}^{\infty}\frac{A^{j}}{j!}$$ where $$A$$ is an $$n \times n$$ matrix

5. If $$A$$ is diagonalizable, $$e^{A}=E\sum_{j=0}^{\infty}\frac{\Lambda^{j}}{j!}E^{-1}$$ 

   where $$E=\begin{bmatrix}... & ... & ... & ... \\ \alpha_1 & \alpha_2 & ... & \alpha_n\\ ... & ... & ... & ...\\\end{bmatrix}, \Lambda=\begin{bmatrix} \lambda_1 & 0 & ... & 0  \\ 0 & \lambda_2 & ... & 0 \\... & ... & ... & ...\\0 & ... & 0& \lambda_{n}\end{bmatrix}, A \alpha_i=\lambda_i\alpha_i$$

------

If $$A$$ is diagonalizable $$(A=E\Lambda E^{-1})$$

$$e^{At}=\sum_{j=0}^{\infty}\frac{(At)^{j}}{j!}$$

$$=\sum_{j=0}^{\infty}\frac{(E\Lambda E^{-1}t)^{j}}{j!}$$

$$=\sum_{j=0}^{\infty}\frac{(E\Lambda tE^{-1})^{j}}{j!}$$

$$=E(\sum_{j=0}^{\infty}(\Lambda t)^{j})E^{-1}$$

$$=Ee^{\Lambda t}E^{-1}$$

------

$$\frac{df}{dt}=f, f(t)=e^t$$

$$\frac{df}{dt}=af,f(t)=e^{at}$$

$$\frac{du}{dt}=A u(t), u(t)=e^{At}u(0)$$

#### Example

![image-20241016155353605](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20241016155353605.png)

$$t=0$$

$$\frac{dv}{dt}=(w-v)+(0-v)$$

$$\frac{dw}{dt}=(v-w)+(0-w)$$

$$u(t)=\begin{bmatrix}v(t)\\w(t)\end{bmatrix}$$

$$\frac{du}{dt}=\begin{bmatrix}\frac{dv}{dt}\\\frac{dw}{dt}\end{bmatrix}$$

$$\frac{dv}{dt}=-2v+w$$

$$\frac{dw}{dt}=v-2w$$

$$\frac{du}{dt}=\begin{bmatrix} -2 & 1 \\ 1 & -2 \end{bmatrix}\begin{bmatrix}v(t)\\w(t)\end{bmatrix}$$

where $$A=\begin{bmatrix} -2 & 1 \\ 1 & -2 \end{bmatrix}, u(t)=\begin{bmatrix}v(t)\\w(t)\end{bmatrix}$$

$$t \rightarrow \infty$$

If $$\frac{du}{dt}=Au(t), u_0=\begin{bmatrix}v_0 \\ w_0 \end{bmatrix}$$

then $$u(t)=e^{At}u_0$$

$$e^{At} : A=\begin{bmatrix}-2 & 1 \\ 1 & -2 \end{bmatrix}$$

$$T=-2+ -2 =-4$$
$$D=4-1=3$$

$$\lambda_1=-1, \lambda=-3$$

$$\lambda_1 : \alpha_1=\begin{bmatrix} 1 \\ 1 \end{bmatrix}, \lambda_2:\alpha_2=\begin{bmatrix} 1 \\ -1 \end{bmatrix}$$

$$E=\begin{bmatrix} \alpha_1 & \alpha_2\end{bmatrix}=\begin{bmatrix}1 & 1 \\ 1 & -1 \end{bmatrix}$$

$$\Lambda=\begin{bmatrix}-1 & 0 \\ 0 & -3 \end{bmatrix}$$

$$E^{-1}=\frac{-1}{2}\begin{bmatrix}-1 & -1 \\ -1 & 1 \end{bmatrix}$$

Recall:

If $$A=\begin{bmatrix} a & b \\ c & d\end{bmatrix}$$

$$A^{-1}=\frac{1}{ad-bc}\begin{bmatrix} d & -b \\ -c & a\end{bmatrix}$$ where $$ad \neq bc$$

$$e^{At}=Ee^{\Lambda t}E^{-1}$$

$$=\begin{bmatrix}1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix}e^{-t} & 0 \\ 0 & e^{-3t} \end{bmatrix} \frac{-1}{2}\begin{bmatrix}-1 & -1 \\ -1 & 1 \end{bmatrix}$$

$$=\begin{bmatrix}1 & 1 \\ 1 & -1 \end{bmatrix} \frac{-1}{2}\begin{bmatrix}-e^{-t} & -e^{-t} \\ -e^{-3t} & e^{-3t} \end{bmatrix}$$

$$=\frac{-1}{2} \begin{bmatrix}-e^{-t}-e^{-3t} & -e^{-t}+e^{-3t} \\ -e^{-t}+e^{-3t} & -e^{-t}-e^{-3t} \end{bmatrix}$$

$$=\frac{1}{2} \begin{bmatrix}e^{-t}+e^{-3t} & e^{-t}-e^{-3t} \\ e^{-t}-e^{-3t} & e^{-t}+e^{-3t} \end{bmatrix}$$

$$u(t)=e^{At}u_0$$

$$=e^{At}\begin{bmatrix}v_0 \\ w_0 \end{bmatrix}$$

------

$$\frac{du}{dt}=Au(t), u(0)=u_0, A$$ is diagonalizable

$$u(t)=e^{At}u_0=Ee^{\Lambda t}E^{-1}u_0$$

$$u(t)=\begin{bmatrix}... & ... & ... & ... \\ \alpha_1 & \alpha_2 & ... & \alpha_n\\ ... & ... & ... & ...\\\end{bmatrix} \begin{bmatrix} e^{\lambda_1 t} & 0 & ... & 0  \\ 0 & e^{\lambda_2 t} & ... & 0 \\... & ... & ... & ...\\0 & ... & 0& e^{\lambda_{n} t}\end{bmatrix} [E^{-1}][u_0]$$

$$=\begin{bmatrix}... & ... & ... & ... \\ \alpha_1 & \alpha_2 & ... & \alpha_n\\ ... & ... & ... & ...\\\end{bmatrix} \begin{bmatrix} e^{\lambda_1 t} & 0 & ... & 0  \\ 0 & e^{\lambda_2 t} & ... & 0 \\... & ... & ... & ...\\0 & ... & 0& e^{\lambda_{n} t}\end{bmatrix} \begin{bmatrix}c_1 \\ c_2\\ ... \\c_n\end{bmatrix}$$

$$=\begin{bmatrix}... & ... & ... & ... \\ \alpha_1 & \alpha_2 & ... & \alpha_n\\ ... & ... & ... & ...\\\end{bmatrix} \begin{bmatrix}c_1 e^{\lambda_1 t} \\ c_2 e^{\lambda_2 t}\\ ... \\c_n e^{\lambda_n t}\end{bmatrix}$$

$$=c_1 e^{\lambda_1 t}\alpha_1+c_2 e^{\lambda_2 t}\alpha_2+...c_n e^{\lambda_n t}\alpha_n$$

where $$A\alpha_i=\lambda_i\alpha_i$$

$$\lim_{t \rightarrow \infty} u(t)$$ Stability

$$u(t)$$ as $$t \rightarrow \infty$$

$$\lambda=a+ib$$

$$e^{\lambda}=e^{a+ib}=e^ae^{ib}$$

$$=e^{a}(\cos(b)+i \sin(b))$$

$$e^{\Lambda t}=e^{at+ibt}=e^{at}(\cos(bt)+i\sin(bt))$$

In general $$\frac{du}{dt}=Au(t)$$

1. If all $$Re(\lambda_i)<0, \lim_{t \rightarrow \infty}e^{\lambda_i t} \rightarrow 0$$

2. If all $$Re(\lambda_i) \leq 0$$ and some $$Re(\lambda_i)=0$$

   $$\lim_{t \rightarrow \infty}e^{\lambda_i t}$$ is bounded

3. If one $$Re(\lambda_i)>0$$, then $$e^{\lambda_i t} \rightarrow \infty$$

Stability

1. Stable System: All $$Re(\lambda_i)<0$$
2. Neutrally Stable System: All $$Re(\lambda_i)\leq 0$$ and $$Re(\lambda_i)= 0$$
3. Unstable System: There is at least one $$Re(\lambda_i)>0$$

------

$$A=\begin{bmatrix}a & b \\ c & d \end{bmatrix}$$

$$\frac{du}{dt}=Au(t)$$

$$D=+, T=- , \lambda_1=-1, \lambda_2=-3$$

$$D=-, T=- , \lambda_1=1, \lambda_2=-3$$

$$D=-, T=+ , \lambda_1=-1, \lambda_2=3$$

If eigenvalues are real, then the stability test is direct

1. The trace $$a+d$$ must be nagetive
2. The determinant $$ad-bc$$ must be positive

------

### Higher order differential equations

$$y'''-3y''+2y'=0$$

$$y'''=\frac{d^3y}{dt^3}$$

$$y''=\frac{d^2y}{dt^2}$$

$$y'=\frac{dy}{dt}$$

$$y'=v$$

$$y''=v'$$

Let $$w=v'=y''$$ then $$w'=v''=y'''=3y''-2y'$$

$$\begin{bmatrix}y' \\ v'\\ w'\end{bmatrix}=\begin{bmatrix}v \\ w\\ 3w-2v\end{bmatrix}=\begin{bmatrix}0 & 1 & 0 \\ 0& 0&1\\ 0& -2&3\end{bmatrix}\begin{bmatrix}y \\ v\\ w\end{bmatrix}$$

------

#### Example

### Coupled differntial equations

$$\frac{dy}{dt}=ay(t),y(t)=e^{at}$$

$$\frac{du}{dt}=Au$$ where  $$A$$ is diagonalizable

change if coordinates: $$\frac{dv}{dt}=\Lambda v$$

$$\begin{bmatrix} \frac{dy_1}{dt} \\ ... \\ \frac{dy_n}{dt}\end{bmatrix}=\begin{bmatrix} a_{11} &...& 0 \\ 0 &...& 0\\ 0 &...& a_{nn}\end{bmatrix}\begin{bmatrix}y_1 \\ ... \\ y_n\end{bmatrix}=\begin{bmatrix}a_{11}y_1(t) \\ ... \\ a_{nn}y_n(t)\end{bmatrix}$$

$$u=Ev$$

$$v=E^{-1}u$$

$$E=\begin{bmatrix}... & ... & ... & ... \\ \alpha_1 & \alpha_2 & ... & \alpha_n\\ ... & ... & ... & ...\\\end{bmatrix}$$

$$A=E\Lambda E^{-1}$$

Show that $$\frac{dv}{dt}=\Lambda v$$

$$\frac{dv}{dt}=\frac{d}{dt}E^{-1}u=E^{-1}\frac{du}{dt}=E^{-1}Au=E^{-1}AE v$$

$$u_0$$ , $$v_0=E^{-1}u_0$$

Solve for $$v(t)\rightarrow u(t)=Ev(t)$$

------

### Complex matrices

Complex number: $$z=a+ib \in \C$$ where $$a,b \in \R$$

$$a$$ is the real part, $$ib$$ is the imaginary part

$$t$$ is $$\R$$, $$t=t+0i$$

$$\sqrt{-1}=i, i^2=-1$$

$$|z^2|=a^2+b^2$$

#### Example

If $$y=1+i$$

then $$|y|^2=1^2+1^2=2$$

$$z^{\ast}=(a+ib)^{\ast}=a-ib$$

$$(z^{\ast})^\ast=((a+ib)^{\ast})^{\ast}=(a-ib)^{\ast}=a+ib$$

If $$z=a+ib$$ and $$w=c+id$$

then

$$z+w=(a+ib)+(c+id)=(a+c)+i(b+d)$$

$$zw=(a+ib)(c+id)=(ac-bd)+i(ad+bc)$$

$$(zw)^{\ast}=z^{\ast}w^{\ast}=(a+ib)^{\ast}(c+id)^{\ast}=(a-ib)(c-id)=(ac-bd)-i(ad+bc)$$

$$(zz^\ast)=|z|^2=(a+ib)(a+ib)=a^2+b^2$$

------

Let $$x \in \C^n$$, then $$x=\begin{bmatrix}x_1 \\ ... \\x_n\end{bmatrix}$$ and $$x_1, ... x_n \in \C$$

$$x^{\ast}=\begin{bmatrix}x_1^{\ast} \\ ... \\x_n^{\ast}\end{bmatrix}$$ conjugate of a vector

Norm:

$$x\in \C^n$$, then $$||x||^2=\sum_{j=1}^{n}|x_j|^2=\sum_{j=1}^{n}x_j^{\ast}x_j=(x^\ast)^Tx$$

Inner product

$$x,y \in \C^n$$

$$x\cdot y=\sum_{j=1}^{n}x_j^{\ast}y_j=(x^\ast)^Ty$$

$$y\cdot x=\sum_{j=1}^{n}y_j^{\ast}x_j=(y^\ast)^Ty$$

$$x \cdot y=(y \cdot x)^{\ast}$$

#### Example

$$x=\begin{bmatrix}1+i\\3i\end{bmatrix}, y=\begin{bmatrix}4 \\ 2-i \end{bmatrix}$$

$$x\cdot y=(x^{\ast})^Ty={\begin{bmatrix}1-i\\-3i\end{bmatrix}}^T\begin{bmatrix}4 \\ 2-i \end{bmatrix}$$

$$=4-4i-6i+3i^2=1-10i$$

$$y\cdot x={\begin{bmatrix}4 \\ 2+i \end{bmatrix}}^T\begin{bmatrix}1+i\\3i\end{bmatrix}$$

$$=4+4i+6i+3i^2=1+10i$$



Let $$c \in \C, x,y \in \C^n$$

$$cx \cdot y=\sum_{j=1}^{n}(cx_j)^{\ast}y_j=\sum_{j=1}^{n}c^\ast x_j^\ast y_j$$

$$=c^\ast \sum_{j=1}^{n}x_j^\ast y_j$$

$$=c^\ast (x \cdot y)$$



$$x \cdot cy=\sum_{j=1}^{n}(x_j)^\ast(cy_j)=\sum_{j=1}^{n}x_j^\ast c y_j$$

$$=c \sum_{j=1}^{n}x_j^\ast y_j$$

$$=c(x \cdot y)$$

------

### Conjugate Transpose

$$A^H$$ Hermitian operator 

Let $$A \equiv [a_{ij}]$$, then

$$A^H=(A^\ast)^T=(A^T)^\ast \equiv [a_{ij}^H]$$ where 

$$a_{ij}^H=a_{ji}^\ast$$

#### Example

$$A=\begin{bmatrix} 1+i & 2-i \\ 3+2i & 4-5i\end{bmatrix}$$

$$A^H=(A^\ast)^T={\begin{bmatrix} 1-i & 2+i \\ 3-2i & 4+5i\end{bmatrix}}^T=\begin{bmatrix} 1-i & 3-2i \\ 2+i & 4+5i\end{bmatrix}$$

Let $$x,y \in \C^n$$ $$x$$ and $$y$$ are orthogonal if and only if $$x \cdot y=0$$

$$x \cdot y$$ or $$x^Hy$$

$$x \cdot y=(x^\ast)^Ty=x^Hy$$ 

------

## Week 8 Session 2

### Outlines

Complex matrices

Fast Fourier Transform

Schur's Decomposition

------

$$K$$

conjugate transpose: $$A^H=(A^{\ast})^T=(A^T)^\ast$$

If $$A \in \R^{m \times n}$$ , $$A^H=(A^\ast)^T=A^T$$

Properties:

1. $$(AB)^H=B^H A^H$$
2. $$(A+B)^H=A^H+B^H$$
3. $$(A^H)^H=A$$
4. $$(A^{-1})^H=(A^H)^{-1}$$ if $$A$$ is invertible
5. $$Det(A^H)=Det(A)^\ast$$
6. $$Tr(A^H)=Tr(A)^\ast$$

------

Hermitian matrices

$$A \in \C^{n \times n}$$ is Hermitian means that $$A^H=A$$

If $$A \in \R^{n \times n}$$, then $$A$$ is Hermitian is and only if $$A$$ is symmetric

$$A=A^T$$

$$A^H=(A^\ast)^T=A^T$$

Thm:

If  $$A \in \C^{n \times n}$$ and $$A=A^H$$ then

1. All its eigenvalues are real

2. Distinct eigenvalues imply mutually orthogonal eigenvectors

Proof:

Claim 1:

If $$A=A^H$$ and $$A \alpha = \lambda \alpha$$ then $$\lambda^\ast = \lambda$$
Proof 1:

$$A \alpha= \lambda \alpha$$
$$\alpha^HA \alpha=\alpha^H \lambda \alpha= \lambda \alpha^H \alpha$$

LHS: $$(\alpha^H A \alpha)^H=\alpha^H A^H (\alpha^H)^H$$

$$=\alpha^H A^H \alpha$$

$$=\alpha^H A \alpha$$

$$=\lambda \alpha^H \alpha$$
RHS: $$(\lambda \alpha^H \alpha)^H=\lambda^{\ast} \alpha^H (\alpha^H)^H$$

$$=\lambda^\ast \alpha^H \alpha$$

$$\lambda \frac{\alpha^H \alpha}{\alpha^H \alpha}=\lambda^\ast \frac{\alpha^H \alpha}{\alpha^H \alpha}$$

$$\lambda=\lambda^\ast$$



Claim 2:

If $$A=A^H$$, $$A\alpha=\lambda_1 \alpha$$, $$A \beta=\lambda_2\beta$$, $$\lambda_1 \neq \lambda_2$$, then $$\alpha^H \beta =0$$
Proof 2:

$$A\alpha=\lambda_1 \alpha$$
$$\beta^H A \alpha=\beta^H \lambda_1 \alpha=\lambda_1 \beta^H \alpha$$

$$A \beta=\lambda_2 \beta$$
$$\alpha^H A \beta=\alpha^H \lambda_2 \beta=\lambda_2\alpha^H \beta$$

LHS: $$(\beta^H A \alpha)^H=\alpha^H A^H (\beta^H)^H$$

$$=\alpha^H A^H \beta$$

$$=\alpha^H A \beta$$

$$=\lambda_2 \alpha^H \beta$$

RHS: $$(\lambda_1\beta^H \alpha)^H= \lambda_1^{\ast} \alpha^H (\beta^H)^H$$

$$=\lambda_1^{\ast}\alpha^H \beta$$

$$=\lambda_1 \alpha^H \beta$$

$$\lambda_2 \alpha^H \beta=\lambda^1 \alpha^H \beta$$

$$(\lambda_2-\lambda_1)\alpha^H \beta=0$$

As $$\lambda_2\neq \lambda_1$$

$$\alpha^H \beta=0$$

------

### Skew Hermitian Matrices

$$A \in \C^{n \times n}$$ and $$A=-A^H$$ or $$(-A=A^H)$$

Thm: 

If $$A \in \C^{n \times n}$$ and $$-A=A^H$$ then

1. All its eigenvalues are imaginary
2. Distinct eigenvalues imply mutually orthogonal eigenvectors

Proof:

Claim 1:

If $$A \in \C^{n \times n}$$ and $$-A=A^H$$ and $$A\alpha=\lambda \alpha$$, then $$\lambda^\ast=-\lambda$$

Proof 1:

$$A\alpha=\lambda \alpha$$
$$\alpha^H A \alpha=\alpha^H \lambda \alpha=\lambda \alpha^H \alpha$$

LHS: $$(\alpha^H A \alpha)^H=\alpha^H A^H (\alpha^H)^H$$

$$=\alpha^H A^H \alpha$$
$$=\alpha^H (-A) \alpha$$
$$=-\alpha^H A \alpha$$
$$=-\alpha^H \lambda \alpha$$
$$=-\lambda \alpha^H \alpha$$
RHS: $$(\lambda \alpha^H \alpha)^H=\lambda^\ast \alpha^H (\alpha^H)^H$$

$$=\lambda^\ast \alpha^H \alpha$$

$$-\lambda \frac{\alpha^H \alpha}{\alpha^H \alpha}=\lambda^\ast \frac{\alpha^H \alpha}{\alpha^H \alpha}$$

$$\lambda^\ast=-\lambda$$



Claim 2:

If $$A \in \C^{n \times n}, A=-A^H, A\alpha=\lambda_1 \alpha, A\beta=\lambda_2 \beta$$ and $$\lambda_1\neq \lambda_2$$ then $$\alpha^H \beta=0$$

Proof 2:

$$A \alpha=\lambda_1 \alpha$$
$$\beta_H A \alpha=\beta_H \lambda_1 \alpha=\lambda_1 \beta^H \alpha$$

$$A\beta=\lambda_2 \beta$$

$$\alpha^H A \beta=\alpha^H \lambda_2 \beta=\lambda_2 \alpha^H \beta$$

LHS: $$(\beta^H A \alpha)^H=\alpha^H A^H (\beta^H)^H$$

$$=\alpha^H A^H \beta$$
$$=\alpha^H(-A)\beta$$
$$=-\alpha^H \lambda_2 \beta$$
$$=-\lambda_2 \alpha^H \beta$$

RHS: $$(\lambda_1 \beta^H \alpha)^H=\lambda_1^\ast \alpha^H (\beta^H)^H$$

$$=\lambda_1^\ast \alpha^H \beta$$
$$=-\lambda_1\alpha^H \beta$$

$$-\lambda_2 \alpha^H \beta=-\lambda_1 \alpha^H \beta$$
$$(\lambda_1-\lambda_2)\alpha^H \beta=0$$

As $$\lambda_1 \neq \lambda_2$$

$$\alpha^H \beta=0$$

------

### Unitary matrix

$$A \in \C^{n \times n}$$ is a unitary matrix means that $$AA^H=A^H A=I$$ $$(A^{-1}=A^H)$$

If $$A$$ is real and unitary, then $$AA^T=A^TA=I$$ $$(A^{-1}=A^T)$$

Thm:

If $$A$$ is a unitary matrix then.

1. All its eigenvalues are on the unit circle $$(\lambda_i)=1$$
2. Distinct eigenvalues imply mutually orthogonal eigenvectors

Proof:

Claim 1:

If $A^{-1}=A^H$ and $$A \alpha=\lambda \alpha$$, then $$|\lambda^2|=\lambda \lambda^\ast=1$$

Proof 1:

$$A\alpha=\lambda \alpha$$
$$\alpha^H A \alpha=\alpha^H \lambda \alpha=\lambda \alpha^H \alpha$$

LHS: $$(\alpha^H A \alpha)^H=\alpha^H A^H (\alpha^H)^H$$

$$=\alpha^H A^H \alpha$$
$$=\alpha^H A^{-1} \alpha$$
$$=\alpha^H \lambda^{-1} \alpha$$

$$=\frac{1}{\lambda}\alpha^H \alpha$$

RHS: $$(\lambda \alpha^H \alpha)^H=\lambda^\ast \alpha^H (\alpha^H)^H$$

$$=\lambda^\ast \alpha^H \alpha$$
$$\frac{1}{\lambda} \frac{\alpha^H \alpha}{\alpha^H \alpha}=\lambda^\ast\frac{\alpha^H \alpha}{\alpha^H \alpha}$$

$$\lambda \lambda^\ast=1$$



Claim 2:

$$x: ||x||^2=x^H x$$

$$y: ux$$ and $$u$$ is a unitary matrix

$$||y||^2=(ux)^Hux=x^Hu^Hux=x^Hx=||x||^2$$

$$u=\begin{bmatrix} ... &... &...& ...\\\ \alpha_1 & \alpha_2 &... & \alpha_n \\ ... &... &...& ...\end{bmatrix}$$

$$\alpha_i \alpha_j =\alpha_i^H\alpha_j=\begin{cases} 0 &\text{if\:} i \neq j\\ 1 &\text{if\:} i = j\end{cases}$$

$$u^Hu=I$$

| Real                       | Complex                    |
| -------------------------- | -------------------------- |
| Symmetric: $$A=A^T$$       | Hermitian: $$A=A^H$$       |
| Skew Symmetric: $$A=-A^T$$ | Skew Hermitian: $$A=-A^H$$ |
| Orthogonal: $$A^T=A^{-1}$$ | Unitary: $$A^H=A^{-1}$$    |

#### Example

$$u=\begin{bmatrix} \cos{\theta} & -\sin{\theta} \\ \sin{\theta} & \cos{\theta}\end{bmatrix}$$ and $$x=\begin{bmatrix}x_1 \\ x_2 \end{bmatrix}$$

$$u=\begin{bmatrix} 1-i & 1+i \\ 1+i & 1-i \end{bmatrix}$$

------

### Discrete Fourier Transform(DFT)

$$x[n], 0\leq n \leq N-1$$

$$x[k], 0 \leq k \leq N-1$$

$$X[k]=\sum_{n=0}^{N-1}x[n] e^{-\frac{i2\pi nk}{N}}$$

$$=\sum_{n=0}^{N-1}x[n]\omega_N^{nk}$$ where $$\omega_N=e^{-\frac{i2 \pi}{N}}$$

$$X[n]=\frac{1}{N}\sum_{k=0}^{N-1}x[k]e^{\frac{i2 \pi nk}{N}}$$

$$=\frac{1}{N} \sum_{k=0}^{N-1}x[k]\omega_N^{-nk}$$

$$X=\begin{bmatrix} 1 & 1 & ... & 1\\ 1 & \omega_N &... & \omega_N^{N-1} \\ ... &...&...&...\\1 & \omega_N^{N-1} &... & \omega_N^{(N-1)(N-1)}\end{bmatrix} \begin{bmatrix}x[0]\\...\\x[N-1]\end{bmatrix}$$

where $$F_N=\begin{bmatrix} 1 & 1 & ... & 1\\ 1 & \omega_N &... & \omega_N^{N-1} \\ ... &...&...&...\\1 & \omega_N^{N-1} &... & \omega_N^{(N-1)(N-1)}\end{bmatrix}$$ and $$x=\begin{bmatrix}x[0]\\...\\x[N-1]\end{bmatrix}$$

$$x=\frac{1}{N} F_N^-1X$$

$$N^2=$$ the number of multiplication operations

$$N(N-1)=$$ the number of addition operations

$$N^2+N^2-N \implies O(N^2)$$

### Fast Fourier Transform

$$X=F_Nx$$

$$N=2^r$$ , $$r \in \Z^{+}$$

$$X=ABCx$$

$$X=\begin{bmatrix}I_{\frac{N}{2}} & D_{\frac{N}{2}} \\ I_{\frac{N}{2}} & -D_{\frac{N}{2}}\end{bmatrix} \begin{bmatrix}F_{\frac{N}{2}} & 0 \\ 0 & -F_{\frac{N}{2}}\end{bmatrix} \begin{bmatrix}\text{even-odd row permutations}\end{bmatrix}x$$

$$\begin{bmatrix}x[0]\\x[1]\\...\\x[N-1]\end{bmatrix} \implies cx \implies \begin{bmatrix}x[0]\\x[2]\\...\\x[N-2]\\x[1]\\x[3]\\...\\x[N-1]\end{bmatrix}$$ where $$\begin{bmatrix}x[0]\\x[2]\\...\\x[N-2]\end{bmatrix}=x'$$ and $$\begin{bmatrix}x[1]\\x[3]\\...\\x[N-1]\end{bmatrix}=x''$$

$$X=\begin{bmatrix}I_{\frac{N}{2}} & D_{\frac{N}{2}} \\ I_{\frac{N}{2}} & -D_{\frac{N}{2}}\end{bmatrix} \begin{bmatrix}F_{\frac{N}{2}} & 0 \\ 0 & F_{\frac{N}{2}}\end{bmatrix} \begin{bmatrix}x' \\ x''\end{bmatrix}$$

$$F_{\frac{N}{2}}=\begin{bmatrix} 1 & 1 & ... & 1\\ 1 & \omega_{\frac{N}{2}} &... & \omega_{\frac{N}{2}}^{\frac{N}{2}-1} \\ ... &...&...&...\\1 & \omega_{\frac{N}{2}}^{\frac{N}{2}-1} &... & \omega_{\frac{N}{2}}^{(\frac{N}{2}-1)(\frac{N}{2}-1)}\end{bmatrix}$$

$$(\frac{N}{2})^2 \equiv$$ number of multiplications

$$(\frac{N}{2})^2 -\frac{N}{2}\equiv$$ number of additions

$$X=\begin{bmatrix}I_{\frac{N}{2}} & D_{\frac{N}{2}} \\ I_{\frac{N}{2}} & -D_{\frac{N}{2}}\end{bmatrix} \begin{bmatrix}F_{\frac{N}{2}} x' \\ F_{\frac{N}{2}}x''\end{bmatrix}$$ where $$\begin{bmatrix}F_{\frac{N}{2}} x'\end{bmatrix}=z'$$ and $$\begin{bmatrix}F_{\frac{N}{2}}x''\end{bmatrix}=z''$$

$$I_{\frac{N}{2}}=\begin{bmatrix} 1 & 0 & ...& 0\\ 0 & 1 &... & 0 \\ ...&...&...&...\\0&...&0&1\end{bmatrix} \in {\frac{N}{2} \times \frac{N}{2}}$$

$$D_{\frac{N}{2}}=\begin{bmatrix} 1 & 0 & ...& 0\\ 0 & \omega_N^1 &... & 0 \\ ...&...&...&...\\0&...&0&\omega_N^{\frac{N}{2}-1}\end{bmatrix}=Diag(1,\omega_N^1, ...\omega_N^{\frac{N}{2}-1} ) \in {\frac{N}{2} \times \frac{N}{2}}$$

$$X=\begin{bmatrix}z'+D_{\frac{N}{2}} z'' \\ z'-D_{\frac{N}{2}}z''\end{bmatrix}$$ where $$\begin{bmatrix}z'+D_{\frac{N}{2}} z''\end{bmatrix}, 0\leq k < \frac{N}{2}$$ and $$\begin{bmatrix}z'-D_{\frac{N}{2}}z''\end{bmatrix}, \frac{N}{2}\leq k \leq N-1$$

$$X[k]=z'[k]+\omega_N^kz''[k]$$ where $$0\leq k < \frac{N}{2}$$

$$X[k]=z'[k-\frac{N}{2}]-\omega_N^{k-\frac{N}{2}}z''[k-\frac{N}{2}]$$ where $$\frac{N}{2}\leq k \leq N-1$$

$$F_{\frac{N}{2}} x'= \begin{bmatrix}I_{\frac{N}{4}} & D_{\frac{N}{4}} \\ I_{\frac{N}{4}} & -D_{\frac{N}{4}}\end{bmatrix} \begin{bmatrix}F_{\frac{N}{4}} & 0 \\ 0 & -F_{\frac{N}{4}}\end{bmatrix} \begin{bmatrix}\text{even-odd row permutations}\end{bmatrix}x'$$

FFT: $$O(N log_2N)$$

------

$$A$$ is diagonalizable if and only if $$A$$ has LIE

Schur's Decomposition

$$A \in \C^{n \times n}$$ is unitarily similar to an upper triangular matrix

$$A=UTU^H$$

where $$U$$ is unitary and $$T$$ is upper triangle

## Week 9 Session 1

### Outlines

Schur's Decomposition

Normal Matrices

------

Thm:
Every $$A \in \C^{n \times n}$$ is unitarily similar to an upper triangular matrix $$T_n$$

$$A=UTU^H$$ where $$U$$ is unitary and $$T$$ is upper triangle

$$A^r=(UTU^H)(UTU^H)...(UTU^H)=UT^rU^H$$

Proof: by mathematical induction

Basis Step: $$n=1$$

$$A=[a]$$

$$A=[1][a][1]^H$$

Induction Step:

Induction hypothesis: If $$A_n \in \C^{n \times n}$$, then $$A$$ is unitarily similar to an upper triangular matrix $$T_n$$

$$A_n=V_n T_n V_n^H$$ where $$V_n^H=V_n^{-1}$$

Claim:

If $$A_{n+1} \in \C^{(n+1)\times {n+1}}$$ then $$A_{n+1}=V_{n+1}T_{n+1} V_{n+1}^H$$ where $$V_{n+1}$$ is unitary and $$T_{n+1}$$ is upper triangular

$$T_{n+1}=V_{n+1}^HA_{n+1}V_{n+1}$$

Proof:

$$\exist x \neq 0: A_{n+1}x=\lambda x$$

$$\exist x_j \neq 0$$ because $$x \neq 0$$

Let $$E=\{e_1,e_2,...,e_{j-1},e_j,e_{j+1},...e_{n+1}\}$$

$$e_1=\begin{bmatrix} 1 \\ 0\\0 \\...\\0\\0\end{bmatrix}$$ , $$e_2=\begin{bmatrix} 0 \\ 1\\0 \\...\\0\\0\end{bmatrix} ... e_{n+1}=\begin{bmatrix} 0 \\ 0\\0 \\...\\0\\1\end{bmatrix}$$

$$L=\{e_1,e_2,...,e_{j-1},x,e_{j+1},...e_{n+1}\}$$

$$L$$ is linearly independent

$$L'=\{x,e_1,e_2,...,e_{j-1},e_{j+1},...e_{n+1}\}$$

Apply Gram-Schmidt to $$L$$ to given $$S$$ and normalize

$$S=\{\alpha_1,\alpha_2,...,\alpha_{j-1},\alpha_j,\alpha_{j+1},...\alpha_{n+1}\}$$

where $$\alpha_1=\frac{x}{||x||}$$

$$\alpha_i \alpha_j=\begin{cases} 1 &\text{if\:} i \neq j \\ 1 & \text{if\:} i =j\end{cases}$$

$$U_{n+1}=\begin{bmatrix} ...&...&...&...\\ \alpha_1& \alpha_2& ... & \alpha_{n+1} \\...&...&...&...\end{bmatrix}$$

$$\widetilde{U_n}=\begin{bmatrix} ...&...&...\\  \alpha_2& ... & \alpha_{n+1} \\...&...&...\end{bmatrix}$$

$$U_{n+1}=\begin{bmatrix} \alpha_1 & \widetilde{U_n}\end{bmatrix}$$

$$U_{n+1}^H=\begin{bmatrix} \alpha_1^H \\ \widetilde{U_n}^H\end{bmatrix}$$

$$U_{n+1}^H A_{n+1} U_{n+1}=\begin{bmatrix} \alpha_1^H \\ \widetilde{U_n}^H\end{bmatrix} A_{n+1} \begin{bmatrix} \alpha_1 & \widetilde{U_n}\end{bmatrix}$$

$$=\begin{bmatrix} \alpha_1^H \\ \widetilde{U_n}^H\end{bmatrix}  \begin{bmatrix} A_{n+1} \alpha_1 & A_{n+1} \widetilde{U_n}\end{bmatrix}$$

$$=\begin{bmatrix}  \alpha_1^H A_{n+1} \alpha_1 &  \alpha_1^H A_{n+1} \widetilde{U_n}\\ \widetilde{U_n}^H A_{n+1} \alpha_1 & \widetilde{U_n}^H A_{n+1} \widetilde{U_n}\end{bmatrix}$$

Say $$b=\alpha_1^H A_{n+1} \alpha_1$$, $$\beta=\widetilde{U_n}^H A_{n+1} \alpha_1$$, $$B_n=\widetilde{U_n}^H A_{n+1} \widetilde{U_n}$$



$$b=\alpha_1^H A_{n+1} \alpha_1$$ 

Dimension: $$(1\times (n+1))((n+1)\times (n+1))((n+1)\times 1)=1 \times 1$$

$$=\alpha_1^H A_{n+1} \frac{x}{||x||}$$

$$=\alpha_1^H \lambda \frac{x}{||x||}$$

$$=\frac{x^H}{||x||} \lambda \frac{x}{||x||}$$

$$=\lambda \frac{x^Hx}{||x||^2}$$

$$=\lambda$$



$$\beta=\widetilde{U_n}^H A_{n+1} \alpha_1$$

$$=\widetilde{U_n}^H A_{n+1} \frac{x}{||x||}$$

$$=\widetilde{U_n}^H \lambda \frac{x}{||x||}$$

$$=\lambda  \widetilde{U_n}^H \alpha_1 $$

$$=\lambda \begin{bmatrix} ... & \alpha_2^H &  ... \\ ... & \alpha_3^H &  ... \\ ...&...&...\\... & \alpha_{n+1}^H &  ...\end{bmatrix} \alpha_1=\mathbf{0}$$



$$U_{n+1}^H A_{n+1} U_{n+1}=\begin{bmatrix} \lambda &  \alpha_1^H A_{n+1} \widetilde{U_n}\\ \ \mathbf{0} & \widetilde{U_n}^H A_{n+1} \widetilde{U_n}\end{bmatrix}$$

$$B_n \in \C^{n \times n}$$

If $$B_n \in \C^{n \times n}$$, then $$B_n=V_nT_nV_n^H$$ as the Induction Hypothesis

where $$V_n$$ is unitary and $$T_n$$ is upper triangular

$$B_n=V_nT_nV_n^H=\widetilde{U_n}^H A_{n+1} \widetilde{U_n}$$

$$T_n=V_n^H  \widetilde{U_n}^H A_{n+1} \widetilde{U_n} V_n$$

$$V_{n+1}=U_{n+1} \begin{bmatrix}1 & \mathbf{0}^H \\ \mathbf{0} & V_n\end{bmatrix}$$

Sub-claim: $$V_{n+1}^H A_{n+1} V_{n+1}=T_{n+1}$$

$$V_{n+1}^H A_{n+1} V_{n+1}=\begin{bmatrix}1 & \mathbf{0}^H \\ \mathcal{0} & V_n^H\end{bmatrix}  U_{n+1}^H A_{n+1} U_{n+1} \begin{bmatrix}1 & \mathbf{0}^H \\ \mathcal{0} & V_n\end{bmatrix} $$

$$=\begin{bmatrix}1 & \mathbf{0}^H \\ \mathbf{0} & V_n^H\end{bmatrix}  \begin{bmatrix} \lambda &  \alpha_1^H A_{n+1} \widetilde{U_n}\\ \ \mathbf{0} & \widetilde{U_n}^H A_{n+1} \widetilde{U_n}\end{bmatrix} \begin{bmatrix}1 & \mathbf{0}^H \\ \mathbf{0} & V_n\end{bmatrix}$$

$$=\begin{bmatrix}1 & \mathbf{0}^H \\ \mathbf{0} & V_n^H\end{bmatrix}  \begin{bmatrix} \lambda &  \alpha_1^H A_{n+1} \widetilde{U_n} V_n\\ \ \mathbf{0} & \widetilde{U_n}^H A_{n+1} \widetilde{U_n}V_n\end{bmatrix} $$

$$=\begin{bmatrix} 1 &  \alpha_1^H A_{n+1} \widetilde{U_n} V_n\\ \ \mathbf{0} & V_n^H\widetilde{U_n}^H A_{n+1} \widetilde{U_n}V_n\end{bmatrix}$$

$$=\begin{bmatrix} 1 &  \alpha_1^H A_{n+1} \widetilde{U_n} V_n\\ \ \mathbf{0} & T_n\end{bmatrix}=T_{n+1}$$

$$T_{n+1}$$ is a upper triangular



$$V_{n+1}^H V_{n+1}=I_{n+1}$$

------

### Schur's Decomposition

$$A \in \C^{n \times n}$$

$$k \in \{1,2,...,n-1\}$$

Initialize: $$T_0=A$$

Step 1:

Define $$A_k \equiv (n-k+1) \times (n-k+1) $$ submatrix in the lower right position of $$T_{k-1}$$

$$T_{k-1}=\begin{bmatrix} ... & ... \\ ... & A_k\end{bmatrix}$$

Step 2:

Find an eigenvalue $$\lambda$$ for $$A_k$$ and its corresponding eigenvector $$x$$

$$A_kx=\lambda x$$

Step 3:

Construct a unitary matrix $$N_k$$ which has its first column as $$\frac{x}{||x||}$$

Use Gram-Schmidt + normalize

$$N_k \equiv$$

Step 4:

For $$k=1$$ , set $$U_1=N_1$$

For $$k>1$$ , $$U_k=\begin{bmatrix} I & 0 \\ 0 & N_k\end{bmatrix}$$

Step 5:

$$T_k=U_k^HT_{k-1}U_k$$

------

#### Example

$$A=\begin{bmatrix} 4 & 0 & 1 \\ 1 & 3 & -1 \\ -1& 0 & 2 \end{bmatrix}$$

Find a Schur's Decomposition of $$A$$

$$k=\{1,2\}$$, $$n=3$$

Initialize

$$T_0=A=\begin{bmatrix} 4 & 0 & 1 \\ 1 & 3 & -1 \\ -1& 0 & 2 \end{bmatrix}$$

$$k=1$$

Step 1:

$$A_1 \equiv (3-1+1) \times (3-1+1)=3 \times 3$$

$$A_1=\begin{bmatrix} 4 & 0 & 1 \\ 1 & 3 & -1 \\ -1& 0 & 2 \end{bmatrix}$$

Step 2:

$$A_1 x=\lambda x$$

$$\lambda=3$$

$$x=\begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$$

$$A_1x=3x$$

Step 3:

$$N_1$$ $$\{e_1,...,x,...,e_n\}$$ $$\rightarrow$$ GS $$\rightarrow$$ Normalize

$$N_1=\begin{bmatrix}0 & 1 & 0 \\ 1 & 0 & 0\\0 & 0 & 1\end{bmatrix}=\begin{bmatrix} \frac{x}{||x||} & \alpha_2 & \alpha_3\end{bmatrix}$$

$$N_1^H N_1=N_1^TN_1=I$$

Step 4:
$$U_1=\begin{bmatrix}0 & 1 & 0\\ 1& 0 & 0\\0&0&1\end{bmatrix}$$

Step 5:

$$T_1=U_1^H T_0 U_1=U_1^T T_0 U_1$$    - $$U_1$$ is real

$$T_1=\begin{bmatrix}0 & 1 & 0\\ 1& 0 & 0\\0&0&1\end{bmatrix}\begin{bmatrix} 4 & 0 & 1 \\ 1 & 3 & -1 \\ -1& 0 & 2 \end{bmatrix}  \begin{bmatrix}0 & 1 & 0\\ 1& 0 & 0\\0&0&1\end{bmatrix}$$

$$=\begin{bmatrix}0 & 1 & 0\\ 1& 0 & 0\\0&0&1\end{bmatrix}\begin{bmatrix} 0 & 4 & 1 \\ 3 & 1 & -1 \\ 0& -1 & 2 \end{bmatrix}$$

$$=\begin{bmatrix}3 & 1 & -1\\ 0& 4 & 1\\0&-1&2\end{bmatrix}$$



$$k=2$$

Step 1:

$$A_2 \equiv (3-2+1) \times (3-2+1)=2 \times 2$$

$$A_2=\begin{bmatrix}4 & 1 \\-1 & 2\end{bmatrix}$$

Step 2:

$$A_2=\begin{bmatrix}4 & 1 \\-1 & 2\end{bmatrix}$$

$$\lambda=3$$

$$x=\begin{bmatrix}1\\-1\end{bmatrix}$$

$$A_2x=3x$$

Step 3:

$$N_2$$

$$E=\{e_1,e_2\}$$

$$L=\{e_1,x\}$$

$$\alpha_1=x=\begin{bmatrix}1\\-1\end{bmatrix}$$

$$\alpha_2=e_1-\frac{e_1 \cdot \alpha_1}{\alpha_1 \cdot \alpha_1}\alpha_1$$

$$=\begin{bmatrix}1\\0\end{bmatrix} - \frac{1}{2} \begin{bmatrix}1\\-1\end{bmatrix}$$

$$=\begin{bmatrix}{\frac{1}{2}}\\{\frac{1}{2}}\end{bmatrix}$$

$$\alpha_1=\begin{bmatrix}1\\-1\end{bmatrix}, \alpha_2=\begin{bmatrix}{\frac{1}{2}}\\{\frac{1}{2}}\end{bmatrix}$$

$$\beta_1=\frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}= \begin{bmatrix}\frac{1}{\sqrt{2}}\\-\frac{1}{\sqrt{2}}\end{bmatrix},\beta_2=\begin{bmatrix}\frac{1}{\sqrt{2}}\\\frac{1}{\sqrt{2}}\end{bmatrix}$$

$$N_2=\begin{bmatrix}\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}\\-\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}\end{bmatrix}$$

Step 4:

$$k \neq 1$$

$$U_k=\begin{bmatrix}I & 0 \\ 0 & N_k\end{bmatrix}$$

$$U_2=\begin{bmatrix} 1 & 0&0\\0& \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}\\0& -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}\end{bmatrix}$$

Step 5:

$$T_2=U_2^HT_1U_2$$

$$=\begin{bmatrix} 1 & 0&0\\0& \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}\\0& \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}\end{bmatrix} \begin{bmatrix}3 & 1 & -1\\ 0& 4 & 1\\0&-1&2\end{bmatrix}\begin{bmatrix} 1 & 0&0\\0& \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}\\0& -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}\end{bmatrix}$$

$$=\begin{bmatrix}3 & \frac{2}{\sqrt{2}} & 0\\ 0& 3 & 2\\0&0&3\end{bmatrix}$$

$$T_2=U_2^HT_1U_2$$

$$=U_2^H U_1^HT_0 U_1 U_2$$

$$=U_2^HU_1^H A U_1 U_2$$

where $$U_2^HV_1^H=V^H, U_1U_2=V$$

If $$A=UTU^H$$ then

$$Det(A-\lambda I)=Det(A-\lambda U U^H)$$

$$=Det(UTU^H-\lambda UIU^H)$$

$$=Det(U(T-\lambda I)U^H)$$

$$=Det(U)Det(T-\lambda I)Det(U^H)$$

$$=Det(T-\lambda I)$$

## Week 9 Session 2

### Outlines

Normal Matrices

Ellipsoid in $$\R^n$$

------

If $$A \in \C^{n \times n}$$, then $$\exist U$$ and $$T$$ such that

$$A=UTU^H$$         - $$(T=U^HAU)$$

where $$U$$ is unitary and $$T$$ is upper triangular

Thm:

$$Det(e^A)=e^{Tr(A)}$$

If $$A$$ is diagonalizable

$$e^A=E\begin{bmatrix}e^{\lambda_1} & ... & 0 \\ 0 & ... & 0 \\ 0& ...&e^{\lambda_n}\end{bmatrix}E^{-1}$$ 

Proof:

$$e^A=\sum_{k=0}^{\infty} \frac{A^k}{k!}$$

$$=\sum_{k=0}^{\infty}\frac{(UTU^H)^k}{k!}$$  - Schur's Decomposition

$$=\sum_{k=0}^{\infty}\frac{UT^kU^H}{k!}$$    - $$U^HU=I$$

$$=U (\sum_{k=0}^{\infty}\frac{T^k}{k!}) U^H$$

Let $$L=\begin{bmatrix} t_{11} & t_{12} & ... & t_{1n} \\ 0 & t_{22} & ... & t_{2n}\\ 0 & 0& ... &...\\0& ...&0 & t_{nn}\end{bmatrix}$$ , $$t_{ij}=0$$ if $$t >j$$

$$X=TT \implies x_{ii}=\sum_{k=1}^{n} t_{ik} t_{ki} = |t_{ii}|^2+\sum_{k=1, k\neq i}^n t_{ik}t_{ki}$$

where $$\sum_{k=1, k\neq i}^n t_{ik}t_{ki}=0$$

$$e^A=U (\sum_{k=0}^{\infty}T^k)U^H$$

$$=U \sum_{k=0}^{\infty} (\frac{1}{k!} \begin{bmatrix} t_{11}^k & \_ & \_ & \_ \\ 0 & t_{22}^k & \_ & \_ \\ 0& ... & ...&\_\\ 0& 0& 0& t_{nn}^k\end{bmatrix})U^H$$

$$=U  (\frac{1}{k!} \begin{bmatrix} \sum_{k=0}^{\infty}t_{11}^k & \_ & \_ & \_ \\ 0 & \sum_{k=0}^{\infty}t_{22}^k & \_ & \_ \\ 0& ... & ...&\_\\ 0& 0& 0& \sum_{k=0}^{\infty}t_{nn}^k\end{bmatrix})U^H$$

$$=U  (\frac{1}{k!} \begin{bmatrix} e^{t_{11}} & \_ & \_ & \_ \\ 0 & e^{t_{22}} & \_ & \_ \\ 0& ... & ...&\_\\ 0& 0& 0& e^{t_{nn}}\end{bmatrix})U^H$$

Let $$\widetilde{T}=\begin{bmatrix} e^{t_{11}} & \_ & \_ & \_ \\ 0 & e^{t_{22}} & \_ & \_ \\ 0& ... & ...&\_\\ 0& 0& 0& e^{t_{nn}}\end{bmatrix}$$

$$e^A=U \widetilde{T}U^H$$

$$Det(e^A)=Det(U \widetilde{T}U^H)$$

$$=Det(U)Det(\widetilde{T})Det(U^H)$$            - $$U^H=U^{-1}$$

$$=Det(\widetilde{T})$$

$$=\prod_{j=1}^{n}e^{t_{jj}}$$

$$=e^{\sum_{j=1}^{n}t_{jj}}$$

$$=e^{Tr(T)}$$

Note:

$$Det(A-\lambda I)=Det(UTU^H-\lambda I)$$

$$=Det(UTU^H-\lambda U I U^H)$$

$$=Det(U(T-\lambda I)U^H)$$

$$=Det(U)Det(T-\lambda I)Det(U^H)$$

$$=Det(T-\lambda I)$$

$$Tr(A)=Tr(T)$$

$$Det(e^A)=e^{Tr(T)}=e^{Tr(A)}$$

------

### Normal Matrices

Definition:
$$A \in \C^{n \times n}$$ is normal if and only if $$AA^H=A^HA$$

Thm:

Spectral Theorem:

$$A \in \C^{n \times n}$$ is normal if and only if $$A$$ is unitarily diagonalizable

$$P:$$ $$A \in \C^{n \times n}$$ is normal

$$Q:$$ $$A$$ is unitarily diagonalizable

$$A^HA=AA^H$$ if and only if $$A=U \Lambda U^H$$

Claim 1: $$P \implies Q$$

If $$A^HA=AA^H$$ then $$A=UTU^H$$

Proof:

$$A=UTU^H$$

$$A^H=(UTU^H)^H=UT^HU^H$$

$$AA^H=UTU^HUT^HU^H$$

$$AA^H=UT U^HU T^H U^H=UT I T^H U^H$$

$$A^HA=UT^HU^HUTU^H=UT^HITU^H$$

$$U^H U TT^H U^HU=U^HUT^HTU^HU$$

$$X=TT^H, Y=T^HT$$

$$X=Y$$

$$TT^H=T^HT$$

$$TT^H=X \equiv [x_{ij}], T^HT=Y \equiv [y_{ij}]$$

$$T=\begin{bmatrix} t_{11} & t_{12} & ... & t_{1n} \\ 0 & t_{22} & ... & t_{2n}\\ 0 & 0& ... &...\\0& ...&0 & t_{nn}\end{bmatrix}, T^H=\begin{bmatrix} t_{11}^\ast & 0 & ... & 0 \\ t_{12}^\ast & t_{22}^\ast & ... & 0\\ ... & ...& ... &0\\t_{1n}^\ast& t_{2n}^\ast&... & t_{nn}^\ast\end{bmatrix}$$

$$x_{ii}=\sum_{k=i}^n t_{ik}t_{ik}^\ast=\sum_{k=i}^n |t_{ik}|^2=\sum_{k=i+1}^n |t_{ik}|^2+|t_{ii}|^2$$

$$ T^H=\begin{bmatrix} t_{11}^\ast & 0 & ... & 0 \\ t_{12}^\ast & t_{22}^\ast & ... & 0\\ ... & ...& ... &0\\t_{1n}^\ast& t_{2n}^\ast&... & t_{nn}^\ast\end{bmatrix},T=\begin{bmatrix} t_{11} & t_{12} & ... & t_{1n} \\ 0 & t_{22} & ... & t_{2n}\\ 0 & 0& ... &...\\0& ...&0 & t_{nn}\end{bmatrix}$$

$$y_{ii}=\sum_{k=1}^i t_{ki}^\ast t_{ki}=\sum_{k=1}^i |t_{ki}|^2=\sum_{k=1}^{i-1} |t_{ki}|^2+|t_{ii}|^2$$

For $$x_{11}=y_{11}$$

$$x_{11}=\sum_{k=2}^n |t_{1k}|^2 +|t_{11}|^2=\sum_{k=1}^{1}|t_{k1}|^2=y_{11}$$

$$\sum_{k=2}^n |t_{1k}|^2 +|t_{11}|^2=|t_{11}|^2$$

$$\sum_{k=2}^n |t_{1k}|^2=0$$

where $$t_{1k} \geq 0$$, then $$t_{12}=t_{13}=...=t_{1n}=0$$

For $$x_{11}=y_{22}$$

$$x_{22}=\sum_{k=3}^{n}|t_{2k}|^2+|t_{22}|^2=\sum_{k=1}^{1} |t_{k2}|^2+|t_{22}|^2$$

$$\sum_{k=3}^{n}|t_{2k}|^2+|t_{22}|^2=|t_{12}|^2+|t_{22}|^2$$

$$\sum_{k=3}^{n}|t_{2k}|^2=0$$

where $$t_{2k} \geq 0$$, then $$t_{22}=t_{23}=...=t_{2n}=0$$

For $$x_{33}=y_{33}$$

$$\sum_{k=4}^{n}|t_{3k}|^2+|t_{33}|^2=|t_{13}|^2+|t_{23}|^2+|t_{33}|^2$$

$$t_{34}=t_{35}=...=t_{3n}=0$$

$$...$$

$$x_{nn}=y_{nn}$$

$$t_{ij} =0$$ for $$i \neq j$$
$$T$$ is a diagonal matrix

------

$$A \in \C^{n \times n}$$

$$\lim_{r \rightarrow \infty} A^r = \mathbf{0}$$

Thm: 

$$A^r \rightarrow \mathbf{0} $$ as $$r \rightarrow \infty$$ if and only if $$|\lambda_i|<1$$ for all eigenvalues of $$A$$

Proof:

$$P$$ if and only if $$Q$$ means

1. $$P \implies Q$$
2. $$Q \implies Q$$

Claim 1:

If $$A^r \rightarrow \mathbf{0}$$ as $$r \rightarrow \infty$$ then $$|\lambda_i|<1$$ for all eigenvalues of $$A$$

$$P \rightarrow Q$$

$$\sim P \rightarrow \sim Q$$

$$\sim Q: $$ Assume that $$\exist \lambda_{j}$$ such that $$|\lambda_{j}|\geq 1$$

$$Ax=\lambda_j x$$                - $$x \neq \mathbf{0}$$

$$A^rx=\lambda_j^r x$$

$$\lim_{r \rightarrow \infty} A^r x= \lim_{r \rightarrow \infty} \lambda_j^r x$$

As $$\lim_{r \rightarrow \infty} \lambda_j^r x \neq 0$$, $$\lim_{r \rightarrow \infty} A^r x \neq 0, x\neq 0$$

$$\lim_{r \rightarrow \infty} A^r \neq 0$$ which is $$\sim P$$

If $$A^r \rightarrow \mathbf{0}$$ as $$r \rightarrow \infty$$ then $$|\lambda_j|<1$$ for all eigenvalues of $$A$$



Claim 2:

If $$|\lambda_j|<1$$ for all eigenvalues of $$A$$ then $$A^r \rightarrow \mathbf{0}$$ as $$r \rightarrow \infty$$

Case 1: $$A$$ has unique eigenvalues

Proof:

Unique eigenvalues $$\implies$$ Linearly independent eigeenvectors

$$A=E \Lambda E^{-1}$$

$$\Lambda=Diag(\lambda_1,...\lambda_n)$$

$$\lim_{r \rightarrow \infty} A^r =\lim_{r \rightarrow \infty} E \Lambda^r E^{-1}$$

$$=E\begin{bmatrix} \lim_{r \rightarrow \infty} {\lambda_1}^r & ... & 0 \\ 0 & ... & 0 \\ 0& ...&\lim_{r \rightarrow \infty} {\lambda_n}^r\end{bmatrix}E^{-1}$$

$$=E\begin{bmatrix} 0 & ... & 0 \\ 0 & ... & 0 \\ 0& ...&0\end{bmatrix}E^{-1}$$               - $$|\lambda_i|<1$$

$$=\mathbf{0}$$

$$A^r=UT^r U^H$$

$$T^r=U^HA^rU$$

$$A^r \rightarrow \mathbf{0}$$ if and only if $$T^r \rightarrow \mathbf{0}$$



$$A^r=E \Lambda^r E^{-1}$$

$$\Lambda^r=E^{-1}A^rE$$

$$A^r \rightarrow \mathbf{0}$$ if and only if $$\Lambda^r \rightarrow \mathbf{0}$$



Case 2: $$A$$ has repeated eigenvalues

Proof:

$$A=UTT^H$$         - Schur's Decomposition

$$|a| \leq b \implies -b \leq |a|\leq b$$

If $$-b \rightarrow 0, b \rightarrow 0$$ then $$a=0$$

$$A=U\begin{bmatrix} t_{11} & t_{12} & ... & t_{1n} \\ 0 & t_{22} & ... & t_{2n}\\ 0 & 0& ... &...\\0& ...&0 & t_{nn}\end{bmatrix}U^H$$

Repeated Eigenvalues

Majorize $$T$$ by $$M$$

$$M=\begin{bmatrix} \mu_1 & \alpha & ... & \alpha \\ 0 & \mu_2 & ... & \alpha\\ 0 & 0& ... &...\\0& ...&0 & \mu_n\end{bmatrix}$$

$$|\lambda_i| \leq \mu_i <1$$

$$\mu_1, \mu_2, ..., \mu_n$$ are unique

$$\lambda_1=0.8, \lambda_2=0.8$$

$$\mu_1=0.8, \mu_2=0.81$$

$$\alpha=\max_{i,j , i \neq j} |t_{ij}|$$

$$|t_{ij}|^r \leq m_{ij}^r$$

$$\lim_{r \rightarrow \infty}M^r = \mathbf{0}$$

$$-m_{ij}^r \leq t_{ij}^r \leq m_{ij}^r$$

$$\lim_{r \rightarrow \infty}M^r = \mathbf{0} \implies \lim_{r \rightarrow \infty}T^r = \mathbf{0}$$

$$\lim_{r \rightarrow \infty}A^r = \lim_{r \rightarrow \infty}UT^rU^H$$

$$=U(\lim_{r \rightarrow \infty}T^r)U^H$$

$$=\mathbf{0}$$



$$T_r=\begin{bmatrix} t_{11}^r & \_ & \_ & \_ \\ 0 & t_{22}^r & ... & \_\\ 0 & 0& ... &...\\0& ...&0 & t_{nn}^r\end{bmatrix}, A \equiv [a_{ij}]$$

$$M=\begin{bmatrix} \mu_1^r & \beta & ... & \beta \\ 0 & \mu_2^r & ... & \beta\\ 0 & 0& ... &...\\0& ...&0 & \mu_n^r\end{bmatrix}, B \equiv [b_{ij}]$$

$$a_{ij} \leq b_{ij}$$

$$\lim_{r \rightarrow \infty}-M^r = \mathbf{0}$$

$$\lim_{r \rightarrow \infty}M^r = \mathbf{0}$$

------

### Ellipsoids in $$\R^n$$

Gaussian Vector

$$X \sim \mathbf{N}(\mu_x,k_{xx})$$

$$f_X(x)=\frac{1}{(2 \pi)^{\frac{n}{2}}(Det(k_{xx}))^\frac{1}{2}} e^{-\frac{(x-\mu _x)^T k_{xx}^{-1}(x-\mu_x)}{2}}$$

$$x^2+y^2=1$$ or $$x_1^2+x_2^2=1$$

$$A =A^T \sim \Lambda (\lambda_1, \lambda_2,...,\lambda_n)$$

$$A=A^H,\lambda_i \in \R$$

Special case: $$\lambda_i >0$$ - Positive Definite

$$\frac{x^2}{a^2}+\frac{y^2}{b^2}=1$$

![image-20241025212155596](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20241025212155596.png)

## Week 10 Session 1

### Outlines

Ellipsoids in $$\R^n$$
Positive Definite Matrices

Rayleigh Quotients: Principal component analysis (PCA)

------

### Principle component analysis (PCA)

This is a linear dimensionality reduction methods that captures maximum variance in the data.

![image-20241030111602043](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20241030111602043.png)

How do we find $$P_1,P_2$$ ?

1. Data and covariance matrix

   Data: $$X: n \times p$$

   $$n:$$ number of samples

   $$p:$$ number of features

   $$j<p$$

   Compute the covariance matrix $$K_{xx}$$

2. Principal components: Find a set of new orthogonal axes (principal components) that maximize variance in the data

   Find $$P_1:$$ This is the direction with the most variance

   Find $$P_2:$$ This is orthogonal to $$P_1$$ and captures the second most variance

3. Rayleigh Quotient and Variance

   Consider $$x$$ such that $$||x||^2=1$$ and $$x$$ represents a direction in the feature space

   Variance of the projected data in the direction of $$x$$ is the Rayleigh quotient

   $$R(x,k)=\frac{x^Tkx}{x^Tx}$$

------

### Ellipsoids in $$\R^n$$

$$A=A^H$$

$$A=A^H \implies \lambda_i \in \R$$

special case: $$\lambda_i>0$$ positive definite

In $$\R^2:\frac{x^2}{a^2}+\frac{y^2}{b^2}=1$$

$$E_2=\{(x,y)\in \R^2:\frac{x^2}{a^2}+\frac{y^2}{b^2}=1\}$$

![image-20241030112149271](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20241030112149271.png)

In $$\R^3: \frac{x^2}{a^2}+\frac{y^2}{b^2}+\frac{z^2}{c^2}=1$$

$$E_3=\{(x,y,z)\in \R^3: \frac{x^2}{a^2}+\frac{y^2}{b^2}+\frac{z^2}{c^2}=1$$

Unit sphere in $$\R^3:a=b=c=1$$

$$S_3=\{(x,y,z)\in \R^3:x^2+y^2+z^2=1\}$$

$$E_n=\{x \in \R^n:||x||^2=1\}$$

$$||x||^2=|x_1|^2+|x_2|^2+...+|x_n|^2$$

$$||x||^2=x^Hx=x^HIx$$

$$x^HAx$$ such that $$A=A^H, \lambda_i>0, \forall i: 1 \leq i \leq n$$

Generalize to: $$x^T Ax$$ such that $$A=A^T, \lambda_i>0, \forall i$$

$$x^TAx=\begin{bmatrix}x_1 & ... & x_n\end{bmatrix} \begin{bmatrix}a_{11} & ... & a_{1n} \\ ... & ... & ...\\a_{n1} & ... & a_{nn}\end{bmatrix} \begin{bmatrix} x_1 \\ ... \\x_n\end{bmatrix}=\sum_{i=1}^{n}\sum_{j=1}^{n}a_{ij}x_ix_j$$

where $$x^T=\begin{bmatrix}x_1 & ... & x_n\end{bmatrix}, A= \begin{bmatrix}a_{11} & ... & a_{1n} \\ ... & ... & ...\\a_{n1} & ... & a_{nn}\end{bmatrix}, x= \begin{bmatrix} x_1 \\ ... \\x_n\end{bmatrix}$$

If $$A=I: \lambda_1=1, \lambda_2=1, e_1=\begin{bmatrix}1 \\0\end{bmatrix}, e_2=\begin{bmatrix}0 \\1\end{bmatrix}$$

Principal axes: Principal axis and the normal vector have the same direction

$$P_i=cn$$ where $$c$$ is a non-zero constant



#### Example

$$A=2 \times 2=I$$

$$x^TIx=x_1^2+x_2^2$$

$$E_2=\{x \in \R^2: x^TIx=1\}$$

$$x_1^2+x_2^2=1$$

$$e_1=P_1=\begin{bmatrix}1 \\0\end{bmatrix}, e_2=P_2= \begin{bmatrix}0 \\1\end{bmatrix}$$



Thm:

Principal axes of ellipsoid $$x^TAx=I$$ are the eigenvector of $$A$$, for real $$A=A^T$$ and $$\lambda>0, \forall i$$

Proof:

$$\phi(x)=x^Tax$$

$$\nabla_x \phi(x)=\nabla_x x^TAx$$

$$=\begin{bmatrix} \frac{\partial\phi(x)}{\partial x_1}\\ ...\\\frac{\partial\phi(x)}{\partial x_n} \end{bmatrix}$$

For any $$k: 1 \leq k \leq n$$

$$\frac{\partial\phi(x)}{\partial x_k}=\frac{\partial}{\partial x_k}x^TAx$$

$$=\frac{\partial}{\partial x_k}\sum_{i=1}^{n}\sum_{j=1}^{n}a_{ij}x_ix_j$$

$$=\sum_{i=1}^{n}\sum_{j=1}^{n}a_{ij} \frac{\partial}{\partial x_k} (x_ix_j)$$

$$=\sum_{i=1}^{n}\sum_{j=1}^{n}a_{ij} (x_i\frac{\partial x_j}{\partial x_k} + x_j\frac{\partial x_i}{\partial x_k})$$

$$=\sum_{i=1}^{n} a_{ik}x_i+\sum_{j=1}^{n}a_{jk}x_j$$           - $$A=A^T$$

$$=2 \sum_{i=1}^n a_{ik}x_i$$

$$\nabla_x \phi(x)==\begin{bmatrix} 2 \sum_{i=1}^n a_{i1}x_i \\ ...\\2 \sum_{i=1}^n a_{in}x_i \end{bmatrix}=2Ax$$

$$2A P_i=cP_i \implies AP_i=\frac{c}{2}P_i$$



Property: If $$A=A^T$$ and $$\lambda_i>0 ,\forall i$$ then the length of principal axis $$P_i$$ is such that 

$$||P_i||=\frac{1}{\sqrt{\lambda_i}}$$

Proof:

$$AP_i=\lambda_i P_i$$

$$E_n=\{x \in \R^n: x^TAx=1\}$$

$$P_i^TAP_i=P_i^T \lambda_i P_i=\lambda_iP_i^TP_i=\lambda_i||P_i||^2$$

$$\lambda_i ||P_i||^2=1$$

$$||P_i||=\frac{1}{\sqrt{\lambda_i}}$$

![image-20241030115939017](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20241030115939017.png)



#### Example

$$2x_1^2-2\sqrt{2}x_1x_2+3x_2^2=1$$
$$\begin{bmatrix}x_1 & x_2\end{bmatrix}\begin{bmatrix} 2 & -2\sqrt{2} \\ -\sqrt{2} & 3\end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}=1$$

$$A=\begin{bmatrix} 2 & -2\sqrt{2} \\ -\sqrt{2} & 3\end{bmatrix}, T=5,D=4$$

$$\lambda=\frac{T \pm \sqrt{T^2-4D}}{2}=\frac{5 \pm \sqrt{5^2-4 \cdot 4}}{2}$$

$$\lambda_1=1,\lambda_2=4$$

$$e_1=\begin{bmatrix} \sqrt{2} \\ 1 \end{bmatrix}, e_2=\begin{bmatrix}-1 \\ \sqrt{2} \end{bmatrix}$$

![image-20241030121823442](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20241030121823442.png)

$$||P_1||=\frac{1}{\sqrt{\lambda_1}}=\frac{1}{1}, ||P_2||=\frac{1}{\sqrt{\lambda_2}}=\frac{1}{\sqrt{4}}=\frac{1}{2}$$

------

### Positive Definite Matrices

$$A=A^H$$

Definition: $$A$$ is positive definite if and only if $$x^HAx>0$$ for all $$x \neq \mathbf{0}$$

Positive Semi-Definite (PSD): $$A$$ is PSD if and only if $$x^HAx \geq 0, \forall x \neq \mathbf{0}$$

Negative Definite (ND): $$A$$ is ND if and only if $$x^HAx < 0, \forall x \neq \mathbf{0}$$

Negative Semi-Definite (NSD): $$A$$ is NSD if and only if $$x^HAx \leq 0, \forall x \neq \mathbf{0}$$

$$x^H Ix = x^Hx=||x||^2 >0$$             - $$I$$ is PD

Thm:

$$A=A^H:A$$ is positive definite if and only if $$\lambda_i>0, \forall i$$

$$P \rightarrow Q, Q \rightarrow P$$

Claim 1:

$$A=A^H:$$ If $$A$$ is positive definite, then $$\lambda_i>0, \forall i$$

Proof: 

$$A \alpha_i=\lambda_i \alpha_i$$      - $$\alpha_i \neq 0$$

$$x^HAx > 0$$        - $$A$$ is positive definite

Let $$u_i=\frac{\alpha_i}{||\alpha_i||}$$

$$u_i^HAu_i=u_i^H\lambda_i u_i=\lambda_i u_i^H u_i=\lambda_i ||u_i||^2=\lambda_i >0$$

QED

Claim 2:

$$A=A^H:$$ If all eigenvalues are positive, then $$x^HAx>0, \forall x \neq 0$$

Proof:

$$A=A^H$$

There exist orthonormal eigenvectors $$u_1,u_2,...u_n$$ (because $$A$$ is a normal matrix) such that 

$$u_i^Hu_j=u_i \cdot u_j=\begin{cases} 0 & \text{if\:} i \neq j \\ 1 & \text{if\:} i=j\end{cases}$$

For any $$x,\exist c_1, ... c_n$$ such that

$$x=\sum_{i=1}^n c_iu_i, x \in \C^n$$

$$Ax=A\sum_{i=1}^n c_iu_i$$

$$=\sum_{i=1}^n c_i A u_i$$

$$=\sum_{i=1}^n c_i \lambda_i u_i$$

$$x^H=(\sum_{i=1}^n c_iu_i)^H=\sum_{i=1}^n c_i^\ast u_i^H$$

$$x^HAx=\sum_{i=1}^n c_i^\ast u_i^H\sum_{j=1}^n c_j \lambda_j u_j$$

$$=\sum_{i=1}^n \sum_{j=1}^n c_i^\ast c_j  \lambda_j u_i^H   u_j$$

$$=\sum_{i=1}^n  c_i^\ast c_j  \lambda_j ||u_i||^2$$

$$=\sum_{i=1}^n  c_i^\ast c_j  \lambda_j >0$$     - $$\lambda_i>0, \exist c_j \neq 0$$

QED

$$A=A^H: A$$ is a PD if and only if $$\lambda_i>0, \forall i$$



Cor: ($$A=A^H)$$

1.  $$A$$ is negative definite if and only if $$\lambda_i<0, \forall i$$
2.  $$A$$ is positive semi-definite if and only if $$\lambda_i\geq0, \forall i$$
3.  $$A$$ is negative semi-definite if and only if $$\lambda_i\leq0, \forall i$$

------

Fact: $$A=A^H:A$$ is positive if and only if all upper left square submatrices $$A_i$$ have $$Det(A_i)>0$$

![image-20241030123419664](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20241030123419664.png)

 

------

### Rayleigh Quotient ($$R(x)$$)

$$A=A^H \implies \lambda_i\in \R$$

Definition: Rayleigh Quotient $$R(x)=\frac{x^HAx}{x^Hx}=\frac{\sum_{i=1}^n\sum_{j=1}^n a_{ij}x_i^\ast x_j}{\sum_{i=1}^n|x_i|^2}$$

$$A$$ is PD

$$\lambda_1\geq \lambda_2 \geq ... \geq \lambda_n >0$$

![image-20241030123706557](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20241030123706557.png)

$$Ax=\lambda x$$

$$x^TAx=x^T\lambda x=\lambda||x||^2$$

$$R(x)=\frac{\lambda||x||^2}{||x||^2}=\lambda$$

## Week 10 Session 2

### Outlines

Rayleigh Quotient

Random Vectors

Whitening

------

### Rayleigh Quotient

$$A=A^H$$

Definition: Rayleigh Quotient $$R(x)=\frac{x^HAx}{x^Hx}=\frac{\sum_{i=1}^n\sum_{j=1}^n a_{ij}x_i^\ast x_j}{\sum_{i=1}^n|x_i|^2}$$

$$A=A^H \implies \lambda_i \in \R \forall i, \lambda_1 \geq \lambda_2 \geq ... \geq \lambda_n$$

For real matrix $$A$$ and real vector $$x: R(x)=\frac{x^TAx}{x^Tx}$$

![image-20241109220655988](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20241109220655988.png)

$$U^TAU=U AU=||U||||AU|| \cos{\theta}$$

------

### Principal component analysis (PCA)

This is a linear dimensionality reduction method that captures the maximum variance in the data.

![image-20241030111602043](C:\Users\Levi\AppData\Roaming\Typora\typora-user-images\image-20241030111602043.png)

How do we find $$P_1,P_2$$ ?

1. Data and covariance matrix

   Data: $$X: n \times p$$

   $$n:$$ number of samples

   $$p:$$ number of features

   $$j<p$$

   Compute the covariance matrix $$K_{xx}$$

2. Principal components: Find a set of new orthogonal axes (principal components) that maximize variance in the data

   Find $$P_1:$$ This is the direction with the most variance

   Find $$P_2:$$ This is orthogonal to $$P_1$$ and captures the second most variance

3. Rayleigh Quotient and Variance

   Consider $$x$$ such that $$||x||^2=1$$ and $$x$$ represents a direction in the feature space

   Variance of the projected data in the direction of $$x$$ is the Rayleigh quotient

   $$R(x,k)=\frac{x^Tkx}{x^Tx}$$

------

### Thm: Rayleigh Principle

If $$A=A^H$$ and $$\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_n$$ for eigenvalues $$\lambda_i$$ then

$$\lambda_1 =\max_{||x||=1} x^HAx$$

and maximum is at $$x=u_1$$ where $$Au_1=\lambda_1 u_1$$

$$A=A^H$$ 

$$\{x \in \C^{n \times 1}: ||x|| =1\}$$

$$\max_{||x||=1} x^HAx=\max_{||x||=1} \frac{x^HAx}{x^Hx}$$

Proof:

 $$A=A^H$$                        - Assumption

$$A$$ is normal                    - $$AA^H=A^HA$$

$$A$$ has orthonormal eigenvectors $$u_1,u_2,..., u_n$$ such that:

$$u_i^H u_j= u_i \cdot u_j=\delta_{ij}=\begin{cases} 0 & \text{if\:} i \neq j \\ 1 & \text{if\:} i=j\end{cases}$$

and $$A u_i = \lambda_i u_i$$

$$\forall x \in \C^{n \times 1}: \exist c_1, c_2, ... ,c_n$$ such that 

$$x= \sum_{i=1}^{n} c_iu_i$$

$$Ax=A \sum_{i=1}^{n} c_iu_i= \sum_{i=1}^{n} c_iAu_i=\sum_{i=1}^{n} c_i \lambda_i u_i$$

$$x^H=(\sum_{i=1}^{n} c_iu_i)^H=\sum_{i=1}^{n}c_i^\ast u_i^H$$

$$x^HAx=\sum_{j=1}^{n} c_i \lambda_i u_i \sum_{i=1}^{n}c_i^\ast u_i^H$$

$$=\sum_{i=1}^{n}  \sum_{j=1}^{n}    c_i^\ast c_i \lambda_i u_i^H u_j$$        - $$u_i^H u_j=\delta_{ij}$$

$$=\sum_{i=1}^{n} |c_i|^2 \lambda_i$$

$$||x||^2=x^Hx=(\sum_{i=1}^{n} c_iu_i)^H(\sum_{i=1}^{n} c_iu_i)$$

$$=\sum_{i=1}^{n}  \sum_{j=1}^{n} c_i^\ast c_i  u_i^H u_j$$

$$=\sum_{i=1}^{n} |c_i|^2 $$

$$x^HAx= \sum_{i=1}^{n} |c_i|^2 \lambda_i$$

$$=\lambda_1 |c_1|^2 +\lambda_2 |c_2|^2 + ... + \lambda_n |c_n|^2$$

$$\leq \lambda_1 |c_1|^2 + \lambda_1 |c_2|^2 + ... \lambda_1 |c_n|^2$$

$$=\lambda_1 \sum_{i=1}^{n}|c_i|^2$$

$$=\lambda_1$$

Therefore, $$x^HAx \leq \lambda_1$$

If $$x=\sum_{i=1}^{n} c_i u_i = u_i + \sum_{j=1, j \neq i}^{n} 0 u_j$$

So $$x^HAx= \sum_{i=1}^{n}|c_i|\lambda_i=\lambda_i$$ if $$x =u_i$$

$$x=u_1 \implies x^HAx=\lambda_1$$

------

Cor:

$$\min_{||x||=1}x^HAx$$

$$\lambda_1= \max_{||x|| \neq 0} \frac{x^HAx}{x^Hx}$$

$$\max_{||x|| \neq 0} \frac{x^HAx}{x^Hx}=\max_{||x|| \neq 0} \frac{x^HAx}{||x||^2}$$

$$=\max_{||x|| \neq 0} (\frac{x}{||x||})^HA(\frac{x}{||x||})$$

$$=\max_{||u||=1}u^HAu=\lambda_1$$

------

Thm:

If $$A=A^H$$ with eigenvalues $$\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_n$$ then

for $$i \geq 2$$, $$\lambda_i=\max_{||x||=1, u_1 \cdot x= u_2 \cdot x=...=u_{i-1} \cdot x=0} x^HAx=0$$

where $$Au_i=\lambda_iu_i$$

$$\lambda_i=\max\{x^HAx: ||x||=1, u_1 \cdot x= u_2 \cdot x=...=u_{i-1} \cdot x=0\}$$

Proof:

Pick $$x \in \C^n$$ such that $$||x||=1$$

$$u_1, u_2, ... u_n$$ are orthonormal eigenvectors of $$A$$
$$x =\sum_{i=1}^{n} c_i u_i$$ for some $$c_1,c_2, ... c_n$$

$$u_j \cdot x=u_j^H \sum_{i=1}^{n}c_i u_i= \sum_{i=1}^{n}c_i u_j^H u_i=c_j$$

$$u_1 \cdot x = u_2 \cdot x= ... = u_{i-1} \cdot x =0$$

$$c_1=c_2=...=c_{i-1}=0$$

$$1=x^Hx=\sum_{j=1}^{n}|c_j|^2=\sum_{j=i}^{n}|c_j|^2$$

$$x^HAx=\sum_{j=1}\lambda_j |c_j|^2$$

$$=\sum_{j=i}^{n} \lambda_j |c_j|^2$$                                    - $$c_1=c_2=...=c_{i-1}=0$$

$$=\lambda_i |c_i|^2 + \lambda_{i+1} |c_{i+1}|^2 + ... + \lambda_n |c_n|^2$$

$$\leq \lambda_i |c_i|^2 + \lambda_{i} |c_{i+1}|^2 + ... + \lambda_i |c_n|^2$$

$$=\lambda_i \sum_{j=i}^{n} |c_j|^2$$

$$=\lambda_i$$

If $$x=u_i$$

$$x=u_i+ \sum_{j \neq i}^n 0 u_j \implies c_i=1, c_j=0, \forall j \neq i$$

$$u_i^HA u_i = u_i^H \lambda_i u_i=\lambda_i u_i^H u_i=\lambda_i$$

### Courant minimax principle

$$\lambda_{i+1}=\min_{v_1, ... v_i, i<n} (\max_{||x||=1, v_1 \cdot x=v_2 \cdot x=...=v_i \cdot x =0} x^HAx)$$

------

### Random Vectors

$$X: \Omega \rightarrow \R$$ such that pullbacks are measurable

Expectation/ Mean $$E[X], \mu_X$$

Variance $$Var[X], \sigma_X^2$$

$$X=\begin{bmatrix} X_1 \\ ... \\X_n\end{bmatrix}$$ where $$X_1, X_n$$ are random variable, $$X$$ is the random vector

$$E[X]=\begin{bmatrix} E[X_1] \\ ... \\ E[X_n]\end{bmatrix}=\begin{bmatrix} \mu_1 \\ ... \\ \mu_n\end{bmatrix}=\mu_X$$

Covariance matrix: $$K_{xx}=E[(X-\mu_X)(X-\mu_X)^T]$$

$$K_{xx}=\begin{bmatrix} \sigma_1^2 & \sigma_{12} & ... & \sigma_{1n} \\ \sigma_{21} & \sigma_2^2 & ... & \sigma_{2n} \\ ... & ... & ... & ...\\ \sigma_{n1} & \sigma_{n2} & ... & \sigma_{n}^2 \end{bmatrix}$$

Say $$\sigma_{2n}=Cov(X_2,X_n)$$

$$K_{xx}$$ is real

$$K_{xx}$$ is symmetric

$$K_{xx}$$ is positive semi-definite

$$K_{xx}=K_{xx}^T$$ means $$Cov(X_i,X_j)=Cov(X_j,X_i), \sigma_{ij}=\sigma_{ji}$$

$$\sigma_i^2 \geq 0$$
Probability density function of a Gaussian vector $$X$$

$$X \sim \text{Gaussian} (\mu_X, K_{xx})$$

$$f_X(x)=\frac{1}{(2 \pi)^{\frac{n}{2}}} (Det(K_{xx}))^{\frac{1}{2}} exp (- \frac{(x-\mu_X)^T K_{xx}^{-1} (X-\mu_X)}{2})$$

Linear transformation of a random vector $$X$$

$$Y=AX$$

$$E[Y]=AE[X]=A\mu_X$$

$$K_{yy}=A K_{xx} A^T$$

### Whitening

Given: $$X$$ has a covariance $$K_{xx}$$, find $$A$$ such that $$Y=AX$$ and $$K_{yy}=I_n$$

$$K_{yy}=I_n \implies Y_1, Y_2, ... Y_n $$ are uncorrelated

$$K_{xx}$$ is positive definite $$\lambda_i >0$$

Thm: 

If $$X \sim K_{xx}$$ and $$Y=AX$$, then $$A= \Lambda^{-\frac{1}{2}}E^T$$

whiten $$K_{xx}: K_{yy}=I$$ where

$$K_{xx} e_i =\lambda_i e_i, \lambda_i>0, \forall i = 1,2,...,n$$

$$E=\begin{bmatrix} e_1 & e_2 & ... & e_n\end{bmatrix}, \Lambda=\begin{bmatrix} \lambda_1 & 0 & ... & 0 \\ 0 & \lambda_2 & ... & 0 \\ ...& ... & ...& ...\\ 0& 0 & ... & \lambda_n\end{bmatrix}$$

$$K_{xx}=E \Lambda E^T$$ spectral theorem and $$K_{xx}$$ is symmetric $$\implies $$ normal

$$E^TE=I$$

Note: If $$A$$ is real, then $$A^T=A^H$$

Proof:

$$K_{yy}=A K_{xx} A^T$$

$$=A E \Lambda E^T A^T$$                      - $$K_{xx}=E \Lambda E^T$$ (spectral theorem)

Let $$A=\Lambda^{-\frac{1}{2}}E^T$$

$$= \Lambda^{-\frac{1}{2}}E^T  E \Lambda E^T(\Lambda^{-\frac{1}{2}}E^T)^T $$

$$=\Lambda^{-\frac{1}{2}} \Lambda \Lambda^{-\frac{1}{2}}$$

$$=I$$
$$\Lambda^{-\frac{1}{2}}=\begin{bmatrix} \lambda_1^{-\frac{1}{2}} & 0 & ... & 0 \\  0 & \lambda_2^{-\frac{1}{2}} & ... & 0 \\ ...& ... & ...& ...\\ 0& 0 & ... & \lambda_n^{-\frac{1}{2}}\end{bmatrix}$$

$$||e_i||=1, e_i \cdot e_j=\delta_{ij}$$

## Week 11 Session 1

### Outlines

Jordan Canonical Form (JCF)

Cholesky Decomposition $$A=LL^H$$

------

### Jordan Canonical Form

Linearly independent eigenvectors: $$A=E \Lambda E^{-1}$$

where $$\Lambda$$ is the diagonal matrix

$$A$$ is similar to $$J$$ where $$J$$ is a diagonal block matrix

$$J=\text{Diagonal}(J_1,J_2,...J_s)$$ where $$J_k$$ is Jordan block for $$k \in \{1,...,s\}$$

$$J=\begin{bmatrix} J_{11} & 0 & 0& ... & 0\\ 0 & J_{22} & 0 & ... & 0\\0& 0& J_{33} & ... & 0\\...& ...& ... & ... & ...\\ 0 & 0 & 0 & ... & J_{nn}\end{bmatrix}\\$$ where $$J_{ii}$$ is a matrix

$$A$$ is similar $$J$$ if and only if $$\exists$$ exists an invertible $$P$$ such that

$$A=PJP^{-1}$$

Jordan Block

$$J_k=\begin{bmatrix} \lambda_k & x & 0& ... & 0\\ 0 & \lambda_k & x & ... & 0\\0& 0& \lambda_k & ... & 0\\...& ...& ... & ... & ...\\ 0 & 0 & 0 & ... & \lambda_k\end{bmatrix}$$

Alternatively

$$J_k=\begin{bmatrix} \lambda_k & 0 & 0& ... & 0\\ x & \lambda_k & 0 & ... & 0\\0& x& \lambda_k & ... & 0\\...& ...& ... & ... & ...\\ 0 & 0 & 0 & ... & \lambda_k\end{bmatrix}$$

$$A \in n \times n$$

Distinct eigenvalues are $$\lambda_1,\lambda_2,...,\lambda_r, r \leq n$$ 

$$P_A(\lambda)=\prod_{i=1}^{r} (\lambda_i-\lambda)^{m_i}$$

$$m_i$$ is the multiplicity for $$\lambda_i$$ and $$\sum_{i=1}^{r}m_i=n$$

$$A \in 6 \times 6$$

$$\lambda_1=\lambda_2=\lambda_3=\lambda_4=3 \implies \lambda_1=3,m_1=4$$

$$\lambda_5=\lambda_6=2 \implies \lambda_2=2,m_2=2$$

$$P_A(\lambda)=(3-\lambda)^4(2-\lambda)^2$$

$$J=\text{Diagonal}(J_1,...,J_s)$$ where $$s$$ is the number of linearly independent eigenvectors

Compute the Jordan Block $$J_k$$

$$m_k \equiv$$ multiplicity of $$\lambda_k$$ for distinct eigenvalues $$\lambda_1,...\lambda_r$$

$$m_k=1: $$ 1 linearly independent eigenvector

$$J_k=[\lambda_k]$$

$$m_k=2:\lambda_k$$ is repeated twice

Case 1: 1 linearly independent eigenvector

$$J_k=\begin{bmatrix} \lambda_k & 1 \\ 0 & \lambda_k \end{bmatrix}$$

Case 2: 2 linearly independent eigenvectors

$$J_k=\begin{bmatrix} \lambda_k & 0 \\ 0 & \lambda_k \end{bmatrix}$$

$$m_k=3:$$

Case 1: 1 linearly independent eigenvector

$$J_k=\begin{bmatrix} \lambda_k & 1 & 0\\ 0 & \lambda_k & 1 \\ 0 & 0 & \lambda_k \end{bmatrix}$$

Case 2: 2 linearly independent eigenvectors

$$J_k=\begin{bmatrix} \lambda_k & 1 & 0\\ 0 & \lambda_k & 0 \\ 0 & 0 & \lambda_k \end{bmatrix}$$

Case 3: 3 linearly independent eigenvectors

$$J_k=\begin{bmatrix} \lambda_k & 0 & 0\\ 0 & \lambda_k & 0 \\ 0 & 0 & \lambda_k \end{bmatrix}$$

$$m_k=4$$

------

$$A=PJP^{-1}$$

#### Example

Let $$A=\begin{bmatrix} 4 & 0 & 0 \\ 0 & 4 & 1 \\ 1 & 0 & 4\end{bmatrix}$$

Find the Jordan decomposition of $$A$$

Find $$J$$ 

$$P_A(\lambda)=Det(A-\lambda I)$$

$$=Det(\begin{bmatrix}4-\lambda & 0 & 0 \\ 0 & 4-\lambda & 1 \\ 1 & 0 & 4-\lambda\end{bmatrix})$$

$$=(4-\lambda)(4-\lambda)(4-\lambda)$$

$$=(4-\lambda)^3$$

$$\lambda_1=\lambda_2=\lambda_3=4$$

$$(A-\lambda I)x =\mathbf{0}$$

$$\begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0\end{bmatrix}x=\mathbf{0}$$

$$x=c \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, c \neq 0$$

$$\text{Dim} (n(A-\lambda I))$$

We have 1 linearly independent eigenvector

$$J=\begin{bmatrix} 4 & 1 & 0 \\ 0 & 4 & 1 \\ 0 & 0 & 4\end{bmatrix}$$

$$A=PJP^{-1}$$

$$AP=PJ$$

$$A\begin{bmatrix} P_1 & P_2 & P_3 \end{bmatrix}=\begin{bmatrix} P_1 & P_2 & P_3 \end{bmatrix}J$$

$$A\begin{bmatrix} P_1 & P_2 & P_3 \end{bmatrix}= \begin{bmatrix} P_1 & P_2 & P_3 \end{bmatrix} \begin{bmatrix} 4 & 1 & 0 \\ 0 & 4 & 1 \\ 0 & 0 & 4\end{bmatrix}$$

$$\begin{bmatrix} AP_1 & AP_2 & AP_3 \end{bmatrix}=\begin{bmatrix} AP_1 & P_1+4P_2 & P_2+4P_3 \end{bmatrix}$$

$$\begin{bmatrix} |& |& |\\P_1 & P_2 & P_3\\ |& |& | \end{bmatrix}\begin{bmatrix} 4 & 1 & 0 \\ 0 & 4 & 1 \\ 0 & 0 & 4\end{bmatrix}$$

$$AP_1=4P_1$$

$$P_1=\begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$$

$$AP_2=P_1+4P_2$$

$$(A-4I)P_2=P_1$$

$$\begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0\end{bmatrix}P_2=\begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$$

$$P_2=\begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

$$AP_3=P_2+4P_3$$

$$(A-4I)P_3=P_2$$

$$\begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0\end{bmatrix}P_3=\begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

$$P_3=\begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$$

$$P=\begin{bmatrix} P_1 & P_2 & P_3 \end{bmatrix}=\begin{bmatrix} 0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0\end{bmatrix}$$

$$P^{-1}=\begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0\end{bmatrix}$$

$$A=PJP^{-1}$$

$$=\begin{bmatrix} 0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0\end{bmatrix} \begin{bmatrix} 4 & 1 & 0 \\ 0 & 4 & 1 \\ 0 & 0 & 4\end{bmatrix} \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0\end{bmatrix}$$

$$=\begin{bmatrix}J_1 & 0 \\ 0 & J_2\end{bmatrix}$$

------

Eigenvalues and eigenvector of diagonal block matrices

$$A=\begin{bmatrix} 2 & -1 & 0 & 0 \\ -1 & 2 & 0 & 0 \\0& 0 & 3 & 1\\ 0 & 0 & 1 &3 \end{bmatrix}$$

$$A_1=\begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}, A_2=\begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$$

$$\lambda_1,\lambda_2, \lambda_3, \lambda_4$$

$$A_1 u_1=\gamma_1 u_1, A_1u_2=\gamma_2u_2, (u_1,u_2)- 2 \times1$$

$$A_2 v_1=\beta_1 v_1, A_2v_2=\beta_2v_2, (v_1,v_2)- 2 \times1$$

$$\lambda_1=\gamma_1$$

$$Ax_1=\lambda_1x_1$$

$$x_1=\begin{bmatrix} u_1 \\ \mathbf{0} \end{bmatrix}$$

$$\lambda_2=\gamma_2$$

$$Ax_2=\lambda_2x_2$$

$$x_2=\begin{bmatrix} u_2 \\ \mathbf{0} \end{bmatrix}$$

$$\lambda_3=\beta_1$$

$$Ax_3=\lambda_3x_3$$

$$x_3=\begin{bmatrix} \mathbf{0}\\ v_1 \end{bmatrix}$$

$$\lambda_4=\beta_4$$

$$Ax_4=\lambda_4 x_4$$

$$x_4=\begin{bmatrix} \mathbf{0}\\ v_2 \end{bmatrix}$$

### Cholesky Decomposition

$$c \in \R$$

$$c=dd, d>0, d \in \R$$

$$A \in n \times n = LL^H$$

$$A=A^H$$ and $$A$$ is positive definite

where $$L$$ - is lower triangular

​                - $$l_{ii}>0,l_{ii} \in \R$$

$$A=LDU$$

where $$L$$ is lower triangular matrix, $$D$$ is the diagonal matrix, $$U$$ is the upper triangular matrix

$$=LD^{1/2}D^{1/2}U$$

$$=LD^{1/2} (U^H(D^{1/2})^{H})^H$$

$$L_1=LD^{1/2}, L_2^H=(U^H(D^{1/2})^{H})^H$$

$$A=\begin{bmatrix} 4 & -2 & -4 \\ -2 & 5 & 8 \\ -4 & 8 & 32\end{bmatrix}$$

$$A=LL^H$$

$$A=A^H$$ because $$A$$ is real and $$A=A^T$$

$$A_{11}=\begin{bmatrix}4\end{bmatrix}, Det(A_{11}) > 0 $$ 

$$A_{22}=\begin{bmatrix}4 & -2 \\ -2 & 5\end{bmatrix}, Det(A_{22})=20-4=16>0$$

$$A_{33}=A: Det(A_{33})=4 [(5)(32)-(8)(8)] +2 [(-2)(32)-(8)(-4)] -4[(-2)(8)-(-4)(5)]>0$$

$$A=\begin{bmatrix} 4 & -2 & -4 \\ -2 & 5 & 8 \\ -4 & 8 & 32\end{bmatrix}$$

$$R_1:R_1$$

$$R_2:R_2+\frac{1}{2}R_1$$

$$R_3:R_3+R_1$$

$$\begin{bmatrix} 4 & -2 & -4 \\ 0 & 4 & 6 \\ 0 & 6 & 28\end{bmatrix}$$

$$R_1:R_1$$

$$R_2:R_2$$

$$R_3:R_3-\frac{3}{2}R_2$$

$$\begin{bmatrix} 4 & -2 & -4 \\ 0 & 4 & 6 \\ 0 & 0 & 9\end{bmatrix}=U$$

$$A=LU$$

$$=\begin{bmatrix} 1 & 0 & 0 \\ -\frac{1}{2} & 1 & 0 \\ -1 & \frac{3}{2} & 1\end{bmatrix}\begin{bmatrix} 4 & -2 & -4 \\ 0 & 4 & 6 \\ 0 & 0 & 9\end{bmatrix}$$

$$=\begin{bmatrix} 1 & 0 & 0 \\ -\frac{1}{2} & 1 & 0 \\ -1 & \frac{3}{2} & 1\end{bmatrix} \begin{bmatrix} 4 & 0 & 0 \\ 0 & 4 & 0 \\ 0 & 0 & 9\end{bmatrix}\begin{bmatrix} 1 & -\frac{1}{2} & -1 \\ 0 & 1 & \frac{3}{2} \\ 0 & 0 & 1\end{bmatrix}$$

where $$L=\begin{bmatrix} 1 & 0 & 0 \\ -\frac{1}{2} & 1 & 0 \\ -1 & \frac{3}{2} & 1\end{bmatrix}, L^T=\begin{bmatrix} 1 & -\frac{1}{2} & -1 \\ 0 & 1 & \frac{3}{2} \\ 0 & 0 & 1\end{bmatrix}$$

$$=\begin{bmatrix} 1 & 0 & 0 \\ -\frac{1}{2} & 1 & 0 \\ -1 & \frac{3}{2} & 1\end{bmatrix} \begin{bmatrix} 2 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3\end{bmatrix} (\begin{bmatrix} 2 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3\end{bmatrix})^T (\begin{bmatrix} 1 & 0 & 0 \\ -\frac{1}{2} & 1 & 0 \\ -1 & \frac{3}{2} & 1\end{bmatrix})^T$$

where $$\begin{bmatrix} 1 & 0 & 0 \\ -\frac{1}{2} & 1 & 0 \\ -1 & \frac{3}{2} & 1\end{bmatrix} \begin{bmatrix} 2 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3\end{bmatrix}=\begin{bmatrix} 2 & 0 & 0\\-1 & 2 & 0\\ -2 & 3 & 3 \end{bmatrix}=L', (\begin{bmatrix} 2 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3\end{bmatrix})^T (\begin{bmatrix} 1 & 0 & 0 \\ -\frac{1}{2} & 1 & 0 \\ -1 & \frac{3}{2} & 1\end{bmatrix})^T=\begin{bmatrix} 2 & -1 & -2\\0 & 2 & 3\\ 0 & 0 & 3 \end{bmatrix}=(L')^T$$

------

### Cholesky Algorithm

$$A=\begin{bmatrix} a_{11} & a_{12} & ...& a_{1n} \\ a_{21} & a_{22} & ... & a_{2n} \\ ... & ... & ... & ...\\a_{n1} & a_{n2} & ... & a_{nn}\end{bmatrix}$$

$$A=A^H$$ and $$A$$ is PD

$$L=\begin{bmatrix} l_{11} & 0 & ... & 0\\ l_{21} & l_{22} & ... & 0 \\ ... & ... & ... & ...\\l_{n1} & l_{n2} & ... & l_{nn}\end{bmatrix}$$

Step 1: Initialization

Set $$l_{ij}=0, \forall i,j : i<j$$

$$l_{11}=\sqrt{a_{11}}$$

$$l_{i1}=\frac{a_{i1}}{l_{11}}$$ for $$ i \in \{2,...,n\}$$

set $$j=2$$

Step 2: If $$j=n+1$$, stop; the algorithm is complete

Otherwise:

Define $$Li'$$ for $$(i=j,j+1,...n)$$ to be a column vector of dimension $$j-1$$ whose components are respectively the first $$j-1$$ elements in the $$i^{th}$$ row of $$L$$

$$L_2'=[]$$

Step 3: Compute $$l_{jj}=\sqrt{a_{jj}-L_j' \cdot L_j'}$$

Step 4: If $$j=n$$ skip to Step 5

Otherwise: $$l_{ij}=\frac{a_{ij}-L_i' \cdot L_j'}{l_{jj}}$$

Step 5: Increase $$j$$ by 1, return to Step 2

#### Example

$$A=\begin{bmatrix} 4 & -2 & -4 \\ -2 & 5 & 8 \\ -4 & 8 & 32\end{bmatrix}$$

Step 1: $$l_{11}=\sqrt{a_{11}}=\sqrt{4}=2$$

$$L=\begin{bmatrix} 2 & 0 & 0 \\ -1 & 2 & 0 \\ -2 & 3 & 2\end{bmatrix}$$

$$l_{21}=\frac{a_{21}}{l_{11}}=\frac{-2}{2}=-1$$

$$l_{31}=\frac{a_{31}}{l_{11}}=\frac{-4}{2}=-2$$

$$j=2$$

Step 2: $$L_2'=\begin{bmatrix}-1\end{bmatrix}, L_3'=\begin{bmatrix}-2\end{bmatrix}$$

Step 3: $$l_{22}=\sqrt{a_{22}-L_2' L_2'}=\sqrt{5-1}=2$$

Step 4: $$j \neq n$$

$$l_{23}=\frac{a_{23}-L_2' L_3'}{l_{23}}=\frac{8-L_2' L_3'}{l_{22}}=\frac{8-(-1)(-2)}{2}=3$$

Step 5: $$j=2+1=3$$

$$j=3$$

Step 2: $$L_i'$$ such that $$i \in \{j,...,n\}$$

$$l_{32}=\frac{a_{32}-L_3' \cdot L_2'}{l_{22}}=\frac{8-(-2)(-1)}{2}=3$$

$$L_3'=\begin{bmatrix} -2 \\ 3 \end{bmatrix}$$

Step 3: $$l_{33}=\sqrt{a_{33}-L_3' \cdot L_3'}=\sqrt{22-13}=3$$

Step 4:

$$j=j+1=4$$

## Week 11 Session 2

### Outlines

Cholesky Algorithm

Singular Value Decomposition (SVD)

Pseudo-inverse

------

### Singular Value Decomposition (SVD)

Let $$A \in \C ^{m \times n}$$, then

$$A=U\Sigma V^H$$

where $$U \in {m \times m}, \Sigma \in {m \times n}, V^H \in {n \times n}$$

where $$U^HU=UU^H=I \in \C^{m \times m}$$    - $$U$$ is unitary

​            $$V^HV=VV^H=I \in \C^{n \times n}$$      - $$V$$ is unitary

$$\Sigma$$ is a semi-diagonal matrix of non-negative real numbers

Ex.

$$\Sigma=\begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}$$

$$\Sigma=\begin{bmatrix} 1 & 0 & 0\\ 0 & 4 & 0\end{bmatrix}$$

The number of non zero numbers in $$\Sigma=\text{Rank}(A)$$

------

How to find $$U, \Sigma, V$$

$$U= \begin{bmatrix} u_1 & u_2 & ... & u_m \end{bmatrix}$$

where $$u_i's$$ are the eigenvectors of $$AA^H$$

$$V=\begin{bmatrix} v_1 & v_2 & ... & v_n \end{bmatrix}$$

where $$v_i's$$ are the eigenvectors of $$A^HA$$

$$u_i's$$ are orthornormal

$$v_i's$$ are orthonormal

Note: $$(A^HA)^H=A^H (A^H)^H=A^H A$$            - $$A^HA$$ is Hermitian

​            $$(AA^H)^H=(A^H)^H A^H=AA^H$$          - $$AA^H$$ is Hermitian

$$A^HA=(U\Sigma V^H)^H (U\Sigma V^H)$$

$$=V \Sigma^H U^H U\Sigma V^H$$                                              - $$U^HU=I$$

$$=V \Sigma^H \Sigma V^H$$

so $$\Sigma^H \Sigma$$ contains the eigenvalues of $$A^HA$$

Similarly

$$AA^H=(U\Sigma V^H)(U\Sigma V^H)^H$$

$$=U\Sigma V^H V \Sigma^H U^H$$                                              - $$V^HV=I$$

$$=U \Sigma \Sigma^H U^H$$

so $$\Sigma \Sigma^H$$ contains the eigenvalues of $$A^HA$$

Since $$\Sigma$$ is a real matrix, $$\Sigma \Sigma^H=\Sigma \Sigma^T$$ and $$\Sigma^H \Sigma=\Sigma^T \Sigma$$

<u>Diagonal</u>: $$\Sigma \Sigma^H$$ and $$\Sigma^H \Sigma$$ contain $$\sigma_1^2, \sigma_2^2, ... , \sigma_r^2$$ which are the eigenvalues of $$A^HA$$ and $$AA^H$$

Note that $$r=\text{Rank}(A)$$

$$\sigma_i's$$ are the singular value

------

Fact: Let $$A=U \Sigma V^H$$

$$\sigma$$ is a singular if and only if $$\exist u \in \C^{m \times 1}$$ and $$\exist v \in \C^{n \times 1}$$ 

such that $$||u||=||v||=1, Av= \sigma u, A^H u =\sigma v$$

Note: Always use this to test the validity of your answer

------

If $$A \in \R^{m \times n}$$, then $$A=U \Sigma V^T$$

where $$U$$ is orthogonal, $$V$$ is orthogonal, and $$\Sigma$$ is real with non-negative numbers

------

#### Example

Let $$A=\begin{bmatrix} 2 & 0 \\ 0 & -3 \\ 0 & 0\end{bmatrix}$$

$$A^TA=\begin{bmatrix} 2 & 0 & 0 \\ 0 & -3 & 0\end{bmatrix}\begin{bmatrix} 2 & 0 \\ 0 & -3 \\ 0 & 0\end{bmatrix}=\begin{bmatrix} 4 & 0 \\ 0 & 9 \end{bmatrix}$$

$$\lambda_1=4, \lambda_2=9$$

$$v_1=\begin{bmatrix} 1 \\ 0 \end{bmatrix},v_2=\begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

$$\sigma_1=\sqrt{4}=2$$

$$\sigma_2=\sqrt{9}=3$$

$$\Sigma=\begin{bmatrix} 2 & 0\\ 0 & 3\\ 0 & 0\end{bmatrix}, V=\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$

$$Av_1=\sigma_1 u_1$$

$$\begin{bmatrix} 2 & 0 \\ 0 & -3 \\ 0 & 0\end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = 2 u_1 = \begin{bmatrix} 2 \\ 0 \\ 0 \end{bmatrix}$$

$$u_1=\begin{bmatrix} 1 \\ 0 \\ 0\end{bmatrix}$$

$$Av_2=\sigma_2u_2$$

$$\begin{bmatrix} 2 & 0 \\ 0 & -3 \\ 0 & 0\end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = 3 u_2 = \begin{bmatrix} 0 \\ -3 \\ 0\end{bmatrix}$$

$$u_2=\begin{bmatrix} 0 \\ -1 \\ 0\end{bmatrix}$$

$$AA^T=\begin{bmatrix} 2 & 0 \\ 0 & -3 \\ 0 & 0\end{bmatrix} \begin{bmatrix} 2 & 0 & 0 \\ 0 & -3 & 0\end{bmatrix}=\begin{bmatrix} 4 & 0 & 0 \\ 0 & 9 & 0\\ 0 & 0 & 0\end{bmatrix}$$

$$\lambda=0$$

$$(AA^T-\lambda I)x =\mathbf{0}$$

$$AA^Tx=\mathbf{0}$$

$$\begin{bmatrix} 4 & 0 & 0 \\ 0 & 9 & 0\\ 0 & 0 & 0\end{bmatrix}x=\mathbf{0}$$

$$x=u_3=\begin{bmatrix} 0 \\ 0 \\ 1\end{bmatrix}$$

$$A=U \Sigma V^H= \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & 0\\ 0 & 0 & 1\end{bmatrix} \begin{bmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0\end{bmatrix} (\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}) ^T$$

------

### Pseudo-Inverse

If $$A \in \C^{m \times n}$$ and $$A^{+}$$ is the pseudo-inverse of $$A$$, then

1. $$AA^{+}A=A$$
2. $$A^{+}AA^{+}=A^{+}$$
3. $$(AA^{+})^H=AA^{+}$$
4. $$(A^{+}A)^H=A^{+}A$$

For $$A= U \Sigma V^H, A^{+}=VD^{-1}U^H$$

$$AA^H=U \Sigma V^H (U \Sigma V^H)^H$$

$$=U \Sigma V^H V \Sigma^H U^H$$

$$=U \Sigma \Sigma^H U^H$$

#### Example

For $$A=\begin{bmatrix} 2 & 0 \\ 0 & -3 \\ 0 & 0\end{bmatrix}$$, Find $$A^{+}$$

$$A=U \Sigma V^H= \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & 0\\ 0 & 0 & 1\end{bmatrix} \begin{bmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0\end{bmatrix} (\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}) ^T$$

$$A^{+}=VD^{-1}U^H=\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} \frac{1}{2} & 0 & 0 \\ 0 & \frac{1}{3} & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & 0\\ 0 & 0 & 1\end{bmatrix}$$

------

$$A=\begin{bmatrix} 4 & 0 \\ -4 & 0 \\ 0 & 1\end{bmatrix}$$

$$A= U \Sigma V^H$$

$$U: AA^H, V: A^HA$$

$$A^HA=A^TA=\begin{bmatrix} 25 & 0 \\ 0 & 1 \end{bmatrix}$$

$$\lambda_1=25, \lambda_2=1$$

$$x_1=\begin{bmatrix} 1 \\ 0 \end{bmatrix}, x_2=\begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

$$\Sigma=\begin{bmatrix} 5 & 0 \\ 0 & 1 \\ 0& 0\end{bmatrix}$$

------

### Schur's Decomposition

$$A=\begin{bmatrix} 4 & 0 & -1 \\ 1 & 3 & -1 \\ -1 & 0 & 2\end{bmatrix}$$

$$A=UTU^H$$

$$n=3$$

$$k \in \{1, ..., n-1\} \implies k\in \{1,2\}$$

Initialize: 

$$T_0=A=\begin{bmatrix} 4 & 0 & -1 \\ 1 & 3 & -1 \\ -1 & 0 & 2\end{bmatrix}$$

$$A_k=T_{k-1}$$

$$k=1$$

Step 1: 

$$A_1=T_0=\begin{bmatrix} 4 & 0 & -1 \\ 1 & 3 & -1 \\ -1 & 0 & 2\end{bmatrix}$$

Step 2: $$\lambda=3$$

$$x= \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} \leftarrow x_2$$

Step 3: $$E=\{e_1,e_2,e_3\}$$

$$E^{-1}=\{e_1, x, e_3\}$$      - Linearly independent

$$e_1=\begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} , e_2= \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, e_3=\begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

$$L=\{x, e_1, e_3\}$$

Apply Gram-Schmidt

$$\beta_1=\alpha_1=x=\begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$$

$$\beta_2=\alpha_2-\frac{\alpha_2 \cdot \beta_1}{\beta_1 \cdot \beta_1}\beta_1=\begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$$

$$\beta_3=\alpha_3-\frac{\alpha_3 \cdot \beta_1}{\beta_1 \cdot \beta_1}\beta_1-\frac{\alpha_3 \cdot \beta_2}{\beta_2 \cdot \beta_2}\beta_2=\begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

$$N_1=\begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0\\ 0 & 0 &1\end{bmatrix}$$

Step 4: $$U_1=N_1=\begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0\\ 0 & 0 &1\end{bmatrix}$$

Step 5: $$T_1=U_1^H T_0 U_1= \begin{bmatrix} 3 & 1 & -1 \\ 0 & 4 & 1\\ 0 & -1 & 2\end{bmatrix}$$

$$k=2$$

Step 1: 

$$A_2=\begin{bmatrix} 4 & 1 \\ -1 & 2 \end{bmatrix}$$

Step 2: 

$$\lambda=3$$

$$x= \begin{bmatrix} 1 \\ -1 \end{bmatrix}$$

Step 3:

$$E=\{e_1,e_2\}$$

$$e_1=\begin{bmatrix} 1 \\ 0 \end{bmatrix},e_2=\begin{bmatrix}0 \\ 1 \end{bmatrix}$$

$$E'=\{e_1,x\}$$ or $$\{x,e_2\}$$

$$L=\{x,e_2\}$$

$$x= \begin{bmatrix} 1 \\ -1 \end{bmatrix}, e_2=\begin{bmatrix}0 \\ 1 \end{bmatrix}$$ 	

Apply Gram-Schmidt

$$\beta_1=\alpha_1=\begin{bmatrix} 1 \\ -1 \end{bmatrix}$$

$$\beta_2=\alpha_2--\frac{\alpha_2 \cdot \beta_1}{\beta_1 \cdot \beta_1}\beta_1=\begin{bmatrix} \frac{1}{2} \\ \frac{1}{2} \end{bmatrix}$$

$$\widetilde{\beta_1}= \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix}=\begin{bmatrix} \frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}} \end{bmatrix},\widetilde{\beta_2}= \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix}$$

$$N_2=\begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}\end{bmatrix}$$

Step 4:

$$U_2=\begin{bmatrix} 1 & \mathbf{0} \\ \mathbf{0} & N_2\end{bmatrix}=\begin{bmatrix} 1 & 0 & 0 \\ 0 & \frac{1}{\sqrt{2}}  &\frac{1}{\sqrt{2}} \\ 0 & -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}\end{bmatrix}$$

Step 5:

$$T_2=U_2^HT_1U_2$$

$$=\begin{bmatrix} 3 & \frac{2}{\sqrt{2}} & 0 \\ 0 & 3  & 2 \\ 0 & 0 & 3\end{bmatrix}$$

$$T_2=U_2^HT_1U_2$$

$$=U_2^HU_1^H T_0 U_1U_2$$

$$=U_2^HU_1^H A U_1U_2$$

$$=(U_1U_2)^H A (U_1U_2)$$

$$U=U_1U_2, T=T_2$$

