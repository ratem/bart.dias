### Directed Acyclic Task Graphs {#sec:taskgraphs}

A *Directed Acyclic (task) Graph (DAG)*, $G=(V,E)$, consists of a set of
*tasks*, $t_i\in V$, which are sequential computations that will not be
analyzed further (sometimes also called *strands*). Tasks are connected
by directed *dependency edges*, $(t_i,t_j)\in E$. An edge $(t_i,t_j)$
means that task $t_j$ is *directly dependent* on task $t_i$ and cannot
be executed before task $t_i$ has completed, for instance, because the
input data for task $t_j$ are produced as output data by task $t_i$. In
general, a task $t_j$ is *dependent* on a task $t_i$ if there is a
directed path from $t_i$ to $t_j$ in $G$. If there is neither a directed
path from $t_i$ to $t_j$ nor a directed path from $t_j$ to $t_i$ in $G$,
the two tasks $t_i$ and $t_j$ are said to be *independent*. Independent
tasks could possibly be executed in parallel, if enough processor-cores
are available, since neither task needs input from nor produces output
to the other. A task $t_i$ may produce data for more than one other
task, so there may be several outgoing edges from $t_i$. Likewise, a
task $t_j$ may need immediate input from more than one task, so there
may be several incoming edges to $t_j$. Since $G$ is acyclic, there is
at least one task $t_r$ in $G$ with no incoming edges; such tasks are
called *root* or *start* tasks. Likewise, there is at least one task
$t_f$ with no outgoing edges. Such tasks are called *final*.

Many computations can be pictured as task graphs. Consider as a first
example the execution of the recursive Quicksort algorithm. The tasks
may be the computations done in pivot selection and partitioning with a
dependency from a pivoting task to the ensuing partitioning task. The
root task will be the initial pivot selection in the input array,
followed by the first partitioning task of the whole array. Dependent
tasks will now be the pivot selection and partitioning of the two
independent parts of the partitioned array and so on and so on. A final
task will depend on all partitioning tasks to have completed and will
indicate that the array has been Quicksorted. We will see later in these
lecture notes how such task graphs suitable for parallel execution can
be generated dynamically as [OpenMP]{.roman} tasks or with Cilk. Another
often encountered type of task DAG is the *fork-join* DAG: A dependent
sequence of fork-join tasks, where each task has a number of dependent,
forked tasks that are all connected to the next join task. A fork-join
DAG is the standard structure of [OpenMP]{.roman} programs corresponding
to a sequence of loops of independent iterations, each of which can be
executed in parallel as a set of forked, independent tasks.

For computations structured as task graphs, there is normally a single
start task taking input of size $O(n)$ and a single, final task
producing the results of the computation. In a dynamic setting, the task
graph typically depends on the input, which can be emphasized by writing
$G(n)$. This $n$ is not to be confused with the number of tasks in $G$.

Each task $t_i$ has an associated amount of work and takes sequential
time $T(t_i)$, typically also depending on $n$. The total amount of work
of a given task graph $G=(V,E)$ with $k$ tasks $t_0,t_1,\ldots,t_{k-1}$
is given by the total time of all tasks and is denoted by
$T_1(n)=\sum_{i=0}^{k-1}T(t_i)$. We will again compare against a best
known sequential algorithm for the problem we are solving, so it holds
that $T_1(n)\geq T_{\mathsf{seq}}(n)$.

Doing a computation as specified by a task graph $G$ sequentially on a
single processor-core amounts to the following: Pick a task $t$ with no
incoming edges and execute it. Remove all outgoing edges $(t,t')$ from
$G$. Continue this process until there are no more tasks in $G$. Since
$G$ is acyclic, there is at least one root task from which the execution
can be started. After execution of this $t$, if $t$ is not the last
task, there will be at least one task with no incoming edges, etc.(if
not, $G$ would not be acyclic). Sequential execution of a task graph,
therefore, amounts to executing the tasks (nodes) in some *topological
order*. Any DAG has a topological order (as can be determined
sequentially in $O(k)$ time steps [@CormenLeisersonRivestStein22]). A
task that has become eligible for execution by having no incoming edges
is said to be *ready*. Since all tasks of $G$ are executed, each task
exactly once, and since there is at least one ready task after
completion of a task, the time taken for the sequential execution is
$O(T_1(n))$.

Imagine that several processor-cores are available. A parallel execution
of a computation specified by a task graph $G$ could proceed as follows:
Pick a ready task. If there is a processor-core that is not busy
executing, assign the task to this core. When a task is completed,
remove all outgoing edges, possibly giving rise to further, ready tasks
(but also possibly not, tasks may have many incoming edges). Continue
this process until there are no more ready tasks. The resulting order of
tasks and assignment to processor-cores is called a *schedule*. The
central property of a schedule is that both dependencies and processor
availability are respected: A task is not executed before all incoming
edges have been removed, which means that dependencies have been
resolved and data been made available to the task; at no time, a
processor-core is assigned more than one task; but at times, cores may
be unassigned and idle.

We are interested in the time taken to execute the work $T_1(n)$ with
some schedule with $p$ processors. This is given by the time for the
last task to finish. We denote the execution time of a (for now not
further specified) $p$ processor schedule by $T_p(n)$ and are, of
course, interested in finding fast schedules.

No matter how scheduling is done, the total amount of work $T_1(n)$ can
never be completed faster than $T_1(n)/p$, the best possible
parallelization. Also, no matter how scheduling is done, tasks that are
dependent on each other must be executed in order. Consider a heaviest
path $(t_r,t_1,\ldots,t_f)$ from the start task $t_r$ to a final task
$t_f$ with the largest amount of total work over the tasks $t_i$ on the
path and define $T\infty(n)=T(t_r)+T(t_1)+\ldots+T(T_f)$ as the work
along such a heaviest path. With sufficiently many processor-cores
available (this number is suggested by $\infty$), indeed a schedule
exists that can achieve running time $T\infty(n)$ (think about this).
Clearly, for any schedule, $T_p(n)\geq T\infty(n)$. These two
observations are often summarized as follows:

-   *Work Law*: $T_p(n)\geq T_1(n)/p\geq T_{\mathsf{seq}}(n)/p$,

-   *Depth Law*: $T_p(n) \geq T\infty(n)$.

The work on a heaviest path in a task graph $G$ is often also called the
*span* or the *depth* of the DAG. A heaviest path is commonly referred
to as a *critical path* with *length* or *weight* $T\infty$. It is also
the parallel time complexity of the DAG.

As an example, consider a fork-join DAG with start and final tasks $t_r$
and $t_f$, with $T(t_r)=1$ and $T(t_f)=1$. The start task forks to a
heavier task $t_1$ with $T(t_1)=4$, and to, say, $27$ light tasks with
one unit of work. All forked tasks join at the final task. Thus,
$T_1(n)=1+4+27+1=33$ and $T\infty(n)=1+4+1=6$. With $p=3$, the Work Law
says that $T_p(n)\geq 33/3=11$ with a (relative) speed-up of at most
$T_p(n)/T_1(n) = 3$ and the Depth Law that $T_p(n)\geq 6$.

With more than, say, $p=10$ processors, the Work law gives a running
time of at least $T_1(n)/p\geq 33/11 = 3$ which is less than
$T\infty(n)=6$ and, therefore, not possible according to the Depth Law.
The maximum speed-up achievable is obviously given by
$T_1(n)/T\infty(n)=33/6=5.5$.

For any schedule, the speed-up is bounded as follows:
$$\mathrm{SU}_{p}(n) = \frac{T_{\mathsf{seq}}(n)}{T^{p}_{\mathsf{par}}(n)} \leq \frac{T_1(n)}{T_p(n)}
  \leq \frac{T_1(n)}{T\infty(n)} \quad .$$

The *parallelism* $\frac{T_1(n)}{T\infty(n)}$ is, therefore, an upper
bound on the achievable relative speed-up and also gives the largest
number of processor-cores for which linear speed-up could be possible.

*Critical path analysis* consisting in finding the longest chain of
dependent, sequential work over all tasks, as used in the Depth Law, is
an important tool to analyze the potential for parallelizing a
computation when thinking of the computation as a task graph. If, for
instance, the critical path $T\infty(n)$ is a constant fraction of
$T_1(n)$, Amdahl's Law applies, which is a sign that a better algorithm
and a better DAG must be found.

We now consider a specific scheduling strategy, so-called *greedy
scheduling*. A greedy scheduler assigns a ready task to an available
processor as soon as possible (task ready and processor available),
meaning that a processor-core is idle only in the case when there is no
ready task. Greedy schedules have a nice upper bound on the achieved
running time, which is captured in the following theorem.

::: {#thm:greedy .theorem}
**Theorem 2.5** (Two-optimality of greedy scheduling). *Let $T_p(n)$ be
the execution time of a DAG $G(n)$ with any *greedy schedule* on $p$
processors, and let $T_p^{*}(n)$ be the execution time with a best
possible $p$ processor schedule. It holds that $$\begin{aligned}
  T_p(n) & \leq & \lfloor T_1(n)/p\rfloor + T\infty(n) \\
  & \leq & 2T^{*}_p(n)
  \quad .
  
\end{aligned}$$*
:::

The proof can be sketched as follows: Divide the work of the scheduler
into discrete steps. A step is called *complete* if all processor-cores
are busy on some tasks and *incomplete* if some cores are idle, which is
the case when there are less ready tasks than processor-cores in that
step. Then, the number of complete steps is bounded by
$\lfloor T_1(n)/p\rfloor$; if there were more, more than the total work
$T_1(n)$ would have been executed. The number of incomplete steps is
bounded by $T\infty(n)$, since each incomplete step reduces the work on
a critical path. The Work and the Depth Law hold for any $p$ processor
schedule, in particular for a best possible schedule, so
$T_1(n)/p\leq T^{*}(n)$ and $T\infty(n)\leq T^{*}(n)$ and the last upper
bound follows. The theorem, therefore, states that the execution time
that can be achieved by a greedy schedule is bounded by two times what
can be achieved by a best possible schedule, a guaranteed
two-approximation!

Neither the definition of greedy schedules nor the theorem says how a
greedy scheduler can or should be implemented. But if it can be shown by
some means that a proposed scheduling algorithm is greedy, the greedy
scheduling theorem says that the running time is within a factor two of
best possible. Greedy scheduling is sometimes called *list scheduling*
and the argument for Theorem [2.5](#thm:greedy){reference-type="ref"
reference="thm:greedy"} is also known as Brent's Theorem as discussed in
Section [2.2.4](#sec:cost-work){reference-type="ref"
reference="sec:cost-work"}. Later in these lecture notes, we will
briefly touch on *work-stealing* which is a decentralized, randomized,
greedy scheduling strategy for certain kinds of DAGs, like the one
explained for Quicksort (called strict, spawn-join
DAG's) [@AroraBlumofePlaxton01].

Some parallel programming models and frameworks make it possible to
dynamically construct what effectively amounts to directed acyclic task
graphs, sometimes with additional structural properties, as the parallel
execution progresses. The run-time system for such frameworks execute a
(greedy) scheduling algorithm using the properties of the task graph.
With the help of Theorem [2.5](#thm:greedy){reference-type="ref"
reference="thm:greedy"}, it is sometimes possible to give provable time
bounds for programs executed on such systems. Examples are
[OpenMP]{.roman} tasks, which will be covered in detail later (see
Section [3.3.13](#sec:omptask){reference-type="ref"
reference="sec:omptask"}), and [Cilk]{.roman}, which we will briefly
touch
upon [@BlumofeJoergKuszmaulLeisersonRandallZhou96; @Leiserson10; @SchardlLee23][^2].

### Loops of Independent Iterations {#sec:looppattern}

Computations are often expressed as loops, in algorithmic pseudo-code
and in real programs. A computation is to be performed for the different
values of the loop iteration variable in the range of this variable,
typically in increasing order of the loop variable:

``` {style="SnippetStyle"}
for (i=0; i<n; i++) {
  c[i] = F(a[i],b[i]);
}
```

In this loop, the iterations (different values of the iteration variable
`i`) are *independent* of each other (provided the function `F` has no
side effects): No computation for iteration $i$ is affected by any
computation for iteration $i'$ before $i$, $i'<i$, and no computation
for a later iteration $i''$, $i''>i$, could possibly affect the
computation for iteration $i$. In such a case, the loop could be
trivially parallelized by dividing the iteration space into $p$ roughly
equal-sized blocks of about $n/p$ iterations and letting each block be
executed by a chosen processor-core.

The assignment of blocks, more generally individual iterations, to
processor-cores is called *loop scheduling* and can be done either fully
explicitly (as sometimes needed when parallelizing with `pthreads`, see
Section [3.2.4](#sec:threadloop){reference-type="ref"
reference="sec:threadloop"}, or with [MPI]{.roman}, see lecture
Block [4.2](#blk:mpi){reference-type="ref" reference="blk:mpi"}) or
implicitly with the aid of a suitable compiler and runtime system by
marking the loop (actually a bad name, since "loop" normally implies
order) as consisting of independent iterations (another misnomer in this
context, "iteration" implies sequential dependency) and, therefore,
parallelizable. An example, which we will see again in much detail
later, is the following [OpenMP]{.roman} style parallelization of a
loop:

``` {style="SnippetStyle"}
#pragma omp parallel for
for (i=0; i<n; i++) {
  c[i] = F(a[i],b[i]);
}
```

With the PRAM model, independent loop computations were handled by
simply assigning a processor to each iteration with the
**par**-construct:

``` {style="SnippetStyle"}
par (0<=i<n) {
  c[i] = F(a[i],b[i]);
}
```

The parallel time of this "loop" on a PRAM would be $O(1)$ steps and the
total number of operations $O(n)$ assuming that each evaluation of the
function $F$ also takes only a constant number of time steps. On a
parallel computer with $p$ processor-cores, optimistically, the parallel
loop can be executed in $\Omega(n/p+1)$ time steps by splitting the $n$
iterations roughly evenly between the $p$ processors. The constant term
is supposed to account for overheads in splitting and assigning the
iterations to the processors. This assumes that also the number of
iterations $n$ is known in advance and that this $n$ is not changed
during the iterations. On parallel computers where the processors are
not operating synchronously in lock-step like in the PRAM, a barrier
synchronization (see Section [2.3.14](#sec:barrier){reference-type="ref"
reference="sec:barrier"}) may be needed after the processor-cores have
finished their iterations in order to ensure that the results in the
`c`-array are all available to all processors. The parallel time of a
"parallel loop" may, thus, have to include the time needed for the
barrier synchronization and will be determined by the slowest
processor-core to finish. Load imbalance could become an issue.

The loop of independent iterations pattern with the function `F` being a
simple, arithmetic-logic expression with the same number of primitive
instructions to be executed independently of the actual argument values
is a standard way of expressing a SIMD parallel computation. One single
stream of instructions controls the computations on multiple data,
namely for all the $n$ inputs of the iteration space. If the
processor-architecture has actual SIMD instructions, the loop of
independent instructions could be a way to instruct the compiler to use
these instructions (see
Section [3.3.16](#sec:simdloops){reference-type="ref"
reference="sec:simdloops"}).

### Independence of Program Fragments

Independent loop iterations, in general, independent program fragments
(which could be tasks as in
Section [2.3.1](#sec:taskgraphs){reference-type="ref"
reference="sec:taskgraphs"}) could possibly be executed concurrently, in
parallel by different, available processor-cores. Independence of
program fragments is a sufficient condition for allowing parallel
execution.

Straightforward conditions for independence of program fragments are the
three *Bernstein conditions* [@Bernstein66]. Let $P_i$ and $P_j$ be two
program fragments, with $P_j$ following after $P_i$ in the program flow.
Each of $P_i$ and $P_j$ has a set of (potential) input variables $I_i$
and a set of (potential) output variables $O_i$. These sets can be
determined statically by program analysis, but whether a potential
output variable will actually be assigned is, in general, undecidable.
The fragments $P_i$ and $P_j$ are *dependent* if either

1.  $O_i\cap I_j\neq\emptyset$ (a *true dependency*, or *flow
    dependency*), or

2.  $I_i\cap O_j\neq\emptyset$ (an anti-dependency), or

3.  $O_i\cap O_j\neq\emptyset$ (an output dependency).

The conditions are obviously sufficient but not necessary: Either may
hold, but input or output may not be read or written by the program
fragment or read or written in some specific order such that the outcome
of the parallel execution is still correct.

Dependencies between the iterations of a loop are called *loop carried
dependencies*, and there are three types, corresponding to the three
Bernstein conditions.

In a *loop carried flow dependency*, the outcome of an earlier iteration
affects the computation of a later iteration:

``` {style="SnippetStyle"}
for (i=k; i<n; i++) {
  a[i] = a[i]+a[i-k];
}
```

Here, the simple computation in iteration $i$ is dependent on output in
variable `a[i-k]` produced in iteration $i-k$ (assuming that $k>0$; for
$k=0$ there would be no such dependency), an earlier iteration if the
iterations were executed in increasing, sequential order. Such
iterations can, therefore, not be done in parallel when expecting a
correct outcome.

In a *loop carried anti-dependency*, the outcome of a later iteration
affects an earlier iteration, if the two iterations were reversed or
carried out simultaneously:

``` {style="SnippetStyle"}
for (i=0; i<n-k; i++) {
  a[i] = a[i]+a[i+k];
}
```

The later iteration $i+k$ updates a variable that is used in iteration
$i$, so if iteration $i+k$ would have been executed before or
concurrently with iteration $i$, the output would be different than
expected from a sequential execution in increasing iteration order and
presumably not be correct.

Finally, in a *loop carried output dependency*, two or more iterations
write to the same output variable(s). If executed simultaneously, the
output would not be well-defined unless the same value is written for
all iterations $i$ (as allowed on the Common CRCW PRAM).

``` {style="SnippetStyle"}
for (i=0; i<n-k; i++) {
  a[0] = a[i];
}
```

This is a first example of a *race condition*, about which we will learn
more in later parts of the lecture notes.

Some loop carried dependencies can be removed by appropriate program
transformations. For instance, the loop carried anti-dependency can be
eliminated by introducing an auxiliary array `b` into which the results
from the computations on array `a` are written:

::::: center
::: minipage
``` {style="SnippetStyle"}
for (i=0; i<n-k; i++) {
  a[i] = a[i]+a[i+k];
}
```
:::

$\longrightarrow$

::: minipage
``` {style="SnippetStyle"}
for (i=0; i<n-k; i++) {
  b[i] = a[i]+a[i+k];
}
```
:::
:::::

The transformed (rewritten) loop now consists of independent iterations,
and, therefore, the iterations can be executed in any order and
concurrently, in parallel. Depending on the surrounding program logic
(where is the result expected?), this may have to be followed by a loop
(of independent iterations) to copy `b` back to `a`, taking $O(n)$
operations, or by a swapping of the two arrays, taking $O(1)$
operations. By similar tricks, sometimes other types of dependencies can
be eliminated.

A *parallelizing compiler* would analyze loops and other constructs for
dependencies and remove dependencies where possible by appropriate
transformations in order to generate code that can exploit a large
number of available processor-cores. Since the dependency problem is in
general undecidable, there is a limit to what such compilers can do.
Results may be modest [@Midkiff12].

### Pipeline {#sec:pipeline}

Consider the following nested loop computation:

``` {style="SnippetStyle"}
for (i=0; i<m; i++) {
  for (j=1; j<n; j++) {
    a[i][j] = a[i][j-1]+a[i][j];
  }
}
```

The inner loop on `j` clearly contains a loop carried flow-dependency
and, therefore, cannot be parallelized without sacrificing correctness
as defined by the sequential loop order. The outer loop on `i` contains
$O(n)$ work per iteration which could be performed in parallel with up
to $m$ processors. The parallel time would be $\Omega(\frac{m}{p} n)$
with up to at most $m$ processors. We write this argument compactly as
$\Omega(\frac{m}{p}n+n)$ parallel time.

A different way of assigning processors to the doubly nested loop work
would be the following: Assume that up to $n$ processor-cores are
available. A processor is assigned for each index $j$ in the inner loop.
The $j$th such processor first sits idle for $j-1$ rounds to wait for
`a[0][j-1]` to have been computed by processors $0,1,\ldots, j-1$ before
$j$. Now, processor $j$ can compute the value `a[0][j]` followed by the
values `a[i][j]` for $i=1,\ldots,m-1$. This latter viewpoint of the
computation is a *linear pipeline*. The parallel running time for
computing all the values in the two-dimensional array can be found by
looking at the last processor $n-1$. This processor will have to wait
for $n-1$ rounds before it can start computing values, after which it
can compute a new value for the remaining $m-1$ elements. This gives a
running time of $O(n+m-1)=O(n+m)$ with $p=n$ processor-cores. For
$p\leq n$ processors the running time can be stated as
$\Omega(\frac{n}{p}m+m+n)$.

The general, linear pipelining pattern assumes that a number of $m$ work
items are to be processed, each requiring work that can be structured
into a sequence of $n$ successive *stages* that have to be carried out
one after the other and each take roughly the same (not necessarily
constant amount of) time. The pipelining pattern allows parallelization
by assigning up to $n$ processors to the individual stages.

Pipelining is a surprisingly versatile technique, which can lead to
highly efficient and fast parallel algorithms for some problems.
Pipelining is, for instance, used in algorithms for data exchange
problems (see Section [4.1.4](#sec:switching){reference-type="ref"
reference="sec:switching"}). Pipelines can be more complex, directed,
acyclic dependency graphs like, for instance, series-parallel graphs.
The essence is that work items pass through the stages of the pipeline,
perhaps being split or combined, and that the parallelism comes from
stages that work in parallel on different work items. The number of
processors that can be employed is thus determined by the number of
pipeline stages and the parallel time by the number of work items.

### Stencil {#sec:stencil}

Here is another frequently occurring nested loop computation. An element
of a two-dimensional $m\times n$ matrix `b[i][j]` is updated with the
result of a constant time computation on a small set of elements of
another matrix `a[i][j]`. In the example here, each update is a simple
average function `avg` on eight elements neighboring `a[i][j]` and
`a[i][j]` itself. The updates in the `b` matrix depend on the `a`
matrix, but by using two matrices, the computation has no loop carried
dependencies. Therefore, both loops could possibly be perfectly
parallelized. Since each element update takes constant time, the total
amount of computation for updating all elements is in $O(mn)$ which with
$p$ processor-cores could ideally be done in $\Theta(mn/p+1)$ time
steps. In a PRAM implementation, a processor is assigned to each matrix
element to do the update; with less than $mn$ processor-cores, the
matrix is thought of as divided into $p$ parts, typically block
submatrices, and a processor-core is assigned to each block to do all
the updates for the block (see
Section [4.2.14](#sec:semanticterms){reference-type="ref"
reference="sec:semanticterms"}
and [4.2.8](#sec:organizingprocesses){reference-type="ref"
reference="sec:organizingprocesses"}). After the update step, the two
matrices are swapped. The computation is repeated until some criteria is
met and the `done`-flag is set to **true**. The pattern is an example of
a two-dimensional, so-called $9$-point *stencil computation*. The
matrices are assumed to have also rows and columns indexed as
`a[-1][j]`, `a[m][j]`, `a[i][-1]` and `a[i][n]`, respectively. This
border of *ghost* rows and columns is sometimes called the *halo*, and
in this example, the halo is of depth one.

As an aside, the code snippet illustrates a handy way of handling
two-dimensional arrays in C (for the best introduction to C,
see [@KernighanRitchie88]). Matrices are stored in row-major order: one
row of consecutive elements (here of type `double`) after the other.
Each matrix is declared as a pointer to an array of rows of $n+2$
elements, and $(m+2)(n+2)$ elements are allocated for each matrix. By
pointer arithmetic, adding one full row and one element, the matrix with
halo can be conveniently addressed. The C compiler can, since $n+2$ is
known (although not static), compute the starting address of each row in
the allocated storage. This will also work for higher-dimensional
matrices as long as the sizes of the lowest, faster changing dimensions
are given in the declaration.

``` {style="SnippetStyle"}
double (*a)[n+2];
double (*b)[n+2];
double (*c)[n+2];
double (*aa)[n+2];
double (*bb)[n+2];

a = (double(*)[n+2])malloc((m+2)*(n+2)*sizeof(double));
aa = a; // save
// and shift address by one row and one column
a = (double(*)[n+2])((char*)a+(n+2+1)*sizeof(double)); 

b = (double(*)[n+2])malloc((m+2)*(n+2)*sizeof(double));
bb = b; // same for b
b = (double(*)[n+2])((char*)b+(n+2+1)*sizeof(double));

int done = 0;
while (!done) {
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      // 9-point stencil
      b[i][j] = avg(a[i][j-1],a[i+1][j-1],a[i+1][j],a[i+1][j+1],
                    a[i][j+1],a[i-1][j+1],a[i-1][j],a[i-1][j-1],
                    a[i][j]);
    }
  }
  c = a; a = b; b = c; // swap matrices a and b
  
  done = ... ; // set when done
}

free(aa); // free as allocated
free(bb);
```

A stencil computation on a $d$-dimensional matrix consists in updating
all matrix elements according to a (most often) constant-time *stencil
rule* that depends on and describes a small, bounded, constant-sized
neighborhood of each matrix element. The total amount of computation per
stencil iteration is then proportional to the size of the
$d$-dimensional matrix. The $9$-point stencil above has as neighbors of
matrix element `a[i][j]` the elements whose distance is at most one in
the maximum metric (Chebyshev distance), which is sometimes called a
Moore-neighborhood. A two-dimensional, $5$-point stencil would have as
neighbors the elements whose Manhattan distance (taxi cab metric) is at
most one. This is sometimes called a von Neumann neighborhood. Both are
examples of first-order stencils; higher order stencils include
neighbors that are farther away in the chosen metric. The stencil rule
above is simply a computation of the average of the nine stencil
elements but could be any other constant-time function, for instance,
the rules of Conway's amazing *Game of Life* [@BerlekampConwayGuy04:4].
In Conway's game, life evolves in a two-dimensional, but potentially
infinite universe. It is an example of a cellular
automaton [@Codd68; @ToffoliMargolus87] and, thus, not strictly a
stencil computation; but a finite universe could easily be imagined and
perhaps still be interesting. The standard use of a $5$-point stencil
computation is Jacobi's method for solving the Poisson differential
equation, where the matrix updates are repeated until
convergence [@Sourcebook03 Chapter 16]. The value in the ghost rows and
columns define the *boundary conditions*. Other, higher-dimensional
stencils, e.g., $27$-point (Chebyshev) or $7$-point (Manhattan) in three
dimensions are also frequently used, as are many other, sometimes also
asymmetric stencils of higher order. Accordingly, there are much
terminology and different notations for stencils in different
application areas.

A single iteration of a one-dimensional, second-order stencil
computation is expressed by the following loop.

``` {style="SnippetStyle"}
for (i=0; i<n; i++) {
  b[i] = a[i-2]+a[i-1]+a[i]+a[i+1]+a[i+2];    
}
```

It can be parallelized to run in $\Theta(n/p+1)$ parallel time with $p$
processor-cores.

### Work pool {#sec:workpool}

A *work pool* maintains items of work to be performed to solve our given
computational problem in a data structure, the *pool*, which makes it
possible to insert and remove items in no particular order. A work item
being solved may give rise to new work items that are inserted into the
pool. The process is repeated until all work items have been processed
and the work pool is empty. Work items can be many things, like tasks
ready for execution, parts of the input, partial results, depending on
the situation. The work pool pattern is clearly attractive for
parallelization. Since there are no dependencies among items in the
pool, several/many processors can conceivably remove and work on work
items from the pool independently. A good parallelization in the sense
of good load balance might be possible if there is at any time during
the computation a sufficient number of work items in the pool (compared
to the number of processor-cores). A non-trivial issue is the *parallel
data structure* needed to allow many processors to concurrently remove
and insert work items into the pool. A *centralized* work pool is
maintained by a single processor, and may be easier to design and reason
about, but may also become a sequential bottleneck for parallelization
when a large number of processors at the same time access the work pool.
A centralized design can easily fall victim to Amdahl's Law. In a
*distributed* work pool, the pool consists in a number of local pools
maintained by the individual processors. As long as there are enough
items in any of the pools, the processors can be kept busy and good load
balance guaranteed. Problems arise when work pools run out of work.
There are two strategies for alleviating the ensuing load balancing
problem. With *work-dealing*, processors whose pools are too full
relative to some (static or dynamic) threshold spontaneously deal out
work item(s) to other processors whose pools may have few(er) items.
With *work-stealing*, processors whose pools have become empty *steal*
work from other pools, and continue stealing until they have either been
successful or until it can be inferred that all pools are empty and the
computation has come to the end. Work-stealing is currently a favored
strategy for which appropriate parallel data structures have been
developed, and where sometimes strong bounds on parallel running time
can be proven. Regardless of what strategy is chosen, the work pool
pattern eventually needs to solve the (distributed) *termination
detection problem*: when is the pool definitely empty?

### Master-worker/Master-slave {#sec:masterworker}

The *master-slave* or *master-worker* pattern is sometimes used to
implement the work pool pattern. A dedicated master processor maintains
a central data structure, from which the slaves or workers are given
work (data to work on, tasks to execute) upon explicit request. The
pattern is often simple to implement but fully centralized, highly
asymmetric, and, thus, easily subject to Amdahl's Law and similar
serialization issues.

### Domain Decomposition

The stencil computation employs a localized, constant time, mostly
position-oblivious update operation to each element of a structured
domain, typically a $d$-dimensional matrix, which is iterated a (large)
number of times until a convergence criterion is met. It appears easy to
parallelize efficiently and can utilize a considerable number of
processor-cores. In the more general *domain decomposition* pattern, a
term which we use here very loosely to characterize a computational
pattern and not necessarily in accordance with terminology from other
domains, the situation is like this: A more or less abstract domain in
$d$-dimensional space over which computations are to be performed is
subdivided into subdomains (not necessarily disjunct) which are assigned
to the available $p$ processors. The work to be done in the subdomains,
say on moving particles, may not be uniformly distributed over the
domain and may possibly move around in the domain. The computation per
work item may or may not be uniform and constant. The computation over
the domain is, like in the stencil pattern, typically to be repeated a
large number of times until convergence.

This pattern generalizes the stencil computation in several respects.
Thus, a static decomposition of the domain may not perform well, since
the subdomains can contain different amounts of work items. Since the
items may move, and since the amount of required computation may change
from iteration to iteration, this pattern will typically need dynamic
load balancing to keep the $p$ processors equally active throughout the
computation.

### Iteration until Convergence, Bulk Synchronous Parallel {#sec:bsppattern}

In both the stencil and the domain decomposition pattern, a parallel
computation is iterated a known or unknown number of times $k$ until
some convergence criterion is met. The parallel time that can be
achieved regardless of the number of processor-cores employed is,
therefore, bounded from below by $\Omega(k)$. If $k$ is large compared
to the total work to be carried out, the achievable speed-up will be
limited by Amdahl's Law.

Another way of looking at the pattern is as follows. Let some number $p$
of processors be given. In each iteration, each processor is assigned a
part of the computation to be done, ideally in such a way that the work
load is evenly balanced over the processors. The processors perform
their work and in cooperation decide whether the termination condition
has been met or not. If not, work for the next iteration is
redistributed over the processors. When the work per iteration is large
compared to the coordination at the end of the iteration consisting in
communication (data exchange) and synchronization, this is a typical
*coarse grained* parallel computation, often referred to as a *Bulk
Synchronous Parallel* (BSP) computation. The term was probably coined by
Les Valiant [@Valiant90; @Bisseling04].

An interesting example that can be cast in the bulk synchronous parallel
pattern is level-wise Breadth-First Search (BFS) in a(n un)directed
graph from some starting vertex. Let $G=(V,E)$ be the given graph with
$n=|V|$ vertices and $m=|E|$ arcs, and $s\in V$ a given source vertex.
The problem is to find the distance from $s$ to all other vertices
$u\in V$ defined as the number of arcs on a shortest path from $s$ to
$u$. A standard, sequential BFS algorithm (see any algorithms
textbook [@CormenLeisersonRivestStein22]) maintains a queue of vertices
being explored in the current iteration and a queue of new vertices to
be explored in the next iteration. It maintains a distance label for
each vertex which is the length of a shortest path from $s$ in the part
of the graph that has been explored so far. Initially, all vertices have
distance label $\infty$, except $s$ which has distance label $0$, since
no part of $G$ has been explored. The invariant to be maintained for
iteration $k,
k=0,1,\ldots$ is that all vertices in the queue of vertices to be
explored have correct distance label $k$. In iteration $k$, all vertices
$u$ in this queue are explored by examining the outgoing arcs $(u,v)\in
E$. If $v$ has a finite distance label already, there is nothing to be
done. If the distance label of $v$ is $\infty$, it is updated to $k+1$
and $v$ is put into the queue of vertices for the next iteration. At the
end of iteration $k$, the two queues are swapped.

It is clear that the algorithm terminates after $K$ iterations where $K$
is the largest finite distance from $s$ of some vertex in $G$. It is
also clear that all arcs are examined at most once. Thus, assuming that
all vertices are reachable from $s$, the complexity of this algorithm is
$\Theta(n+m)$ if the queue operations are in $O(1)$.

There is much potential for parallelization. Vertices in the queue of
vertices to be explored in iteration $k$ can be processed in parallel
since order is not important and all arcs out of such vertices can also
be examined in parallel; provided that vertices and arcs are available
to the processor-cores and that conflicts, for instance, when inserting
vertices into the queue of vertices for the next iteration can be
handled. By the end of an iteration, arcs and vertices may have to be
exchanged between processor-cores and queues consolidated for the next
iteration. In the best possible case, a parallel running time in
$\Omega(\frac{n+m}{p}+K)$ could be possible. If $m$ is large compared to
$K$ and perhaps $n$, that is if $G$ is not sparse and has low diameter,
there might be enough "bulk" work for the processor-cores so that
reasonable speed-up can be achieved in practice.

### Data Distribution {#sec:datadistributions}

Parallel algorithms working on structured data often seek to split the
data into (disjoint) parts on which processor-cores can work
independently in an embarrassingly parallel fashion. We have seen this
approach with the stencil pattern and will see it again many times. The
splitting of the data can be explicit by reorganizing the data into
disjoint parts accessible to the available processors; or implicit by
providing naming schemes and transformations to conveniently access the
different parts. Structured data are here thought of as arrays, vectors,
matrices, higher-dimensional matrices, etc., of objects that can
themselves be structured. We refer to the splitting of such objects as
*data distribution*, which may be an active operation to be performed
(repeatedly) during the execution of a parallel algorithm or a matter of
providing means to refer to the parts of the data in the required
fashion.

Let some linear data structure of $n$ elements be given, e.g., an array,
and let $b, b\geq 1$ be a chosen *block size*. Let $p$ be the number of
processors, numbered from $0$ to $p-1$. In a *block cyclic* data
distribution the $n$-element array is split into *blocks* of consecutive
elements of $b$ elements each; one last piece may have fewer than $b$
elements, depending on whether $b$ divides $n$. Number these blocks
consecutively starting from $0$. Then blocks $0,p,2p,\ldots$ can or will
be accessed by processor $0$, blocks $1,p+1,2p+1,\ldots$ by processor
$1$, blocks $2,p+2,2p+2,\ldots$ by processor $2$, in general blocks
$i,p+i,2p+i,\ldots$ by processor $i, 0\leq i<p$.

A *cyclic* data distribution is the special case where $b=1$. A
*blockwise* data distribution is the special case where roughly $b=n/p$
and where rounding is done such that, as far as possible, each processor
has a block of at least one element.

A higher-dimensional matrix may likewise be divided into smaller blocks
(many possibilities) and distributed in a block cyclic way. Special
cases for two-dimensional matrices are the *row-wise* distribution where
each processor is assigned to work on a consecutive number of full rows
of the matrix, and the *column-wise* distribution where each processor
is assigned to work on a consecutive number of full columns of the
matrix. We will see examples of the use of such distribution in
Section [4.2.30](#sec:la){reference-type="ref" reference="sec:la"}.

### Compaction, Gather and Scatter {#sec:arraycompaction}

Consider the loop below. A marker array `mark[i]` is given, which for
each element of the array `a[i]`, tells whether the element is to be
kept or not for some later computation. The marked elements are copied
into the `b[j]` array in the loop order of appearance.

``` {style="SnippetStyle"}
j = 0;
for (i=0; i<n; i++) {
  if (mark[i]) b[j++] = a[i];
}
```

As we will see in the rest of these lectures, this pattern, which is
called *array compaction*, is important and surprisingly versatile.
Unfortunately, the loop has obvious dependencies, e.g., the increment of
`j`, and so far we have no means of parallelizing it.
Section [2.4.5](#sec:prefixsums){reference-type="ref"
reference="sec:prefixsums"} will be devoted to this problem.

The (dense) *gather* and *scatter* patterns rearrange array elements and
are illustrated below. Given an index array `ix[i]` with values
$0\leq\texttt{ix[i]}<n$ which is not necessarily required to be a
permutation (it may be that `ix[i]==ix[j]` for some, different `i` and
`j`), the gather pattern copies elements from `b` in the order given by
the index array into `a`. The scatter pattern is the opposite and copies
into the `a` array in index order from the `b` array in sequential loop
order.

``` {style="SnippetStyle"}
// gather
for (i=0; i<n; i++) {
  a[i] = b[ix[i]];
}
// scatter
for (i=0; i<n; i++) {
  b[ix[i]] = a[i];
}
```

Ideally, with $p$ processor cores, both of the patterns can be
parallelized to run in $\Theta(n/p+1)$ parallel time steps. Dependent on
the index array, there may be concurrent reading in the gather pattern.
If implemented on a PRAM, this pattern requires either concurrent read
capabilities or prior knowledge that the index array is indeed a
permutation. Likewise, the scatter pattern may incur concurrent writing.
If implemented on a PRAM, sufficiently strong concurrent write
capabilities are required depending on which values may be written. More
liberal gather and scatter patterns would allow the `b`-array to be an
$m$-element array with $m\geq n$ and indices in this array.

### Data Exchange, Collective Communication {#sec:exchangepatterns}

Different parts of the data being processed at different stages of the
execution of a parallel algorithm may be managed by or have special
affinity to different processor-cores; indeed, this was the case for
many of the parallel patterns discussed above. It can, therefore, be
convenient or even necessary (as will be seen in
Chapter [4](#chp:distributedmemory){reference-type="ref"
reference="chp:distributedmemory"}) to explicitly exchange or reorganize
data between processor-cores at different stages of the computation.
Such *exchange patterns* and operations are frequent in Parallel
Computing and are also often referred to as *collective communication*
or just *collectives* because all affected processor-cores jointly take
part in and jointly, by appropriate underlying algorithms, effect the
exchange.

The following exchange and reduction patterns are traditionally
considered.

-   *Broadcast* and all-to-all broadcast, in which one, or all,
    processors have data to be distributed to all other processors.

-   *Gather*, in which a specific processor collects individual data
    from all other processors.

-   *Scatter*, in which a specific processor has individual data to be
    transmitted to each of the other processors.

-   *All-to-all*, in which all processors have specific, individual data
    to each of the other processors.

-   *Reduction*, reduction-broadcast (all-reduce) and reduction-scatter,
    in which data are combined together under an associative operator,
    with the results stored either at a specific processor, at all
    processors, or distributed in parts over all processors.

All these patterns are explicitly found in, for instance, [MPI]{.roman}
and will be discussed in great detail in
Section [4.2.28](#sec:collective){reference-type="ref"
reference="sec:collective"}; but they do appear explicitly and
implicitly in many other Parallel Computing contexts as well.

### Reduction, Map-reduce and Scan

Surprisingly many problems can be viewed as reduction problems: A
(large) number of input values which can be numbers, vectors, matrices,
complex objects (texts, pictures, data bases) etc. are combined together
using an associative, functional rule to arrive at the solution. Subsets
of elements can be assigned to processor-cores, and by associativity,
the reduction can be performed by repeated reduction of disjoint pairs
of sets of values. The reduction pattern is a well parallelizable design
pattern. The pattern can be made more powerful and flexible by allowing,
for instance, a precomputation on the input values before reduction;
this operation is often, for instance, in functional programming, called
a *map operation*, and the combined pattern has been popularized as
*map-reduce* [@DeanGhemawat08; @DeanGhemawat10]. Many variations are
possible and have been proposed.

A related pattern is the *scan* or prefix sums where the associative
rule is applied on the input values in sequence: The result for the
$i$th input is the associative reduction of all inputs before, up to and
possibly including input $i$. A scan computes these prefix sums for all
$i$ [@Blelloch89]. We will see many applications of prefix sums and the
scan operation throughout these lectures, see
Section [2.4.5](#sec:prefixsums){reference-type="ref"
reference="sec:prefixsums"} also for efficient algorithms for computing
the scan.

### Barrier Synchronization {#sec:barrier}

Some of the patterns above divide a computation into separate stages
that are executed one after the other, for instance, until some
convergence criterion is met. Other patterns and computations assume
that computations done by other processors have been completed before a
processor can continue to its next phase of computation. Ensuring such
requires some form of *synchronization* between processor-cores and is
the task of what we here call the *barrier* parallel pattern. Semantic
barrier synchronization means that a processor that has reached a
certain point in its computation, called the *barrier* (point), is not
allowed to proceed before all other processors have reached their
barrier point. After the barrier synchronization, updates and
computations performed by the other processors shall be available to the
processor to the extent that this is required.

Barrier synchronization can be implicit or explicit; for arguing for the
correctness of a particular parallel algorithms it is often required,
though, to know at which points the processors are synchronized in the
sense of all having reached a certain point and having a consistent view
of the computation.

In a lock-step, synchronized model like the PRAM, explicit barrier
synchronization is not (or rarely) needed. In asynchronous,
shared-memory models and systems, various forms of barriers are needed
to ensure correctness, and they are typically provided. The
[OpenMP]{.roman} thread model, see
Section [3.3](#sec:openmpframework){reference-type="ref"
reference="sec:openmpframework"}, provides implicit barrier points as
well as explicit barrier synchronization constructs. Unlike for the
PRAM, barriers are typically not for free. In asynchronous models,
barrier synchronization of $p$ processors takes $\Theta(\log p)$
parallel time steps. Many interesting standard algorithms for barrier
synchronization on non-synchronous (non-PRAM) shared-memory systems can
be found in [@MellorCrummeyScott91].

In distributed memory models, the required synchronization is sometimes
guaranteed by the semantics of the provided communication operations,
whether implicit or explicit. Explicit semantic barrier constructs may
be provided as well, although they may be needed less often (to preview:
in [MPI]{.roman}, an explicit barrier is almost never needed!). Also
here, $\Theta(\log p)$ dependent parallel operations go into enforcing a
semantic barrier.


