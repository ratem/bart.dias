#Lectures on Parallel Computing, Jesper Larsson Träff, TU Wien, June 30th, 2024

## Second block (1-2 lectures)

The bar for Parallel Computing is high. We judge parallel algorithms and
implementations by comparing them against the *best possible* sequential
algorithm or implementation for solving the given computational problem,
and in cases where the best possible (lower bound) is not known, against
the *best known* sequential algorithm or implementation. The reasoning
is that we, by using the dedicated parallel resources at hand, want to
improve over what we can already do with a sequential algorithm on our
system. With our parallel machine, we want to solve problems faster
and/or better on some account.

For now, our parallel model and system will be left unspecified. Some
number $p$ of processor-cores interact to solve the problem at hand.

### Sequential and Parallel Time {#sec:timecomplexity}

Parallel Computing is both a theoretical discipline and a
practical/experimental endeavor. As a theoretical discipline, Parallel
Computing is interested in the performance of algorithms in some models
(RAM, PRAM, and more realistic settings), and typically looks at the
performance in the worst possible case (worst possible inputs) when the
input size is sufficiently large. Let [Seq]{.sans-serif} and
[Par]{.sans-serif} denote sequential and parallel algorithms for a
problem we are interested in solving. The parallel algorithm, in
contrast to the sequential algorithm, additionally specifies how
processors are to be employed in the solution, how they interact and
coordinate, and how they exchange information. The sequential and
parallel algorithms may be "similar" in idea and structure; they may
also, as we have already seen
(Theorem [2.1](#alg:fastmax){reference-type="ref"
reference="alg:fastmax"}), be completely different. This is fine as long
as we can argue or even prove that they both correctly solve the given
problem.

By $T_{\mathsf{seq}}(n)$ and $T^{p}_{\mathsf{par}}(n)$ we denote the
running times (depending on how our model accounts for time, for
instance, number of steps taken) of [Seq]{.sans-serif} and
[Par]{.sans-serif} on worst-case inputs of size $n$ with one processor
for the sequential algorithm [Seq]{.sans-serif} and with $p$
processor-cores for the parallel algorithm [Par]{.sans-serif}. The best
possible and best known algorithms for solving a given problem are those
with the best worst-case asymptotic complexities. For a given problem,
the best possible sequential running time is often denoted as
$T^{*}(n)$, a function of the input size $n$ [@JaJa92; @RauberRunger13],
which then defines the *sequential complexity* of the given problem. In
the same way, we can define the *parallel time complexity* $T\infty(n)$
for a given parallel algorithm [Par]{.sans-serif} as the smallest
running time that this algorithm can achieve using sufficiently many
processors. The number of processors to use to achieve this best running
time can then be turned into a function of the input size $n$. If the
parallel algorithm is the *fastest possible* algorithm for our given
problem, $T\infty(n)$ is the parallel time complexity of the problem.

As always, constants do matter(!), but they will often be ignored here
and hidden behind $O, \Omega, \Theta, o, \omega$. Recall the definitions
and rules for manipulating such expressions, see for
instance [@CormenLeisersonRivestStein22] or any other algorithms text,
and note that, for parallel algorithms, the worst-case time is a
function of two variables, problem size $n$ and number of
processor-cores $p$. Saying that some $T^{p}_{\mathsf{par}}(n)$ is in
$O(f(p,n))$ then means that

$$\exists C>0, \exists N,P>0: \forall n\geq N, p\geq P:
  0\leq T^{p}_{\mathsf{par}}(n)\leq C f(p,n)$$ and that some
$T^{p}_{\mathsf{par}}(n)$ is in $\Theta(f(p,n))$ that
$$\exists C_0, C_1>0, \exists N,P>0: \forall n\geq N, p\geq P: 0\leq
  C_0 f(p,n)\leq T^{p}_{\mathsf{par}}(n)\leq C_1 f(p,n) \quad .$$

We may sometimes let the number of processors $p$ change as a function
of the problem size, $p=f(n)$ ("What is the best number of processors
for this problem size?" as in the definition of parallel time
complexity), or the problem size change as a function of the number of
processors, $n=g(p)$ ("What is a good problem size for this number of
processors?"), in which case the asymptotics are of one variable.

Typical sequential, best known/best possible worst-case complexities for
some of our computational problems are [@CormenLeisersonRivestStein22]:

-   $\Theta(\log n)$: Searching for an element in an ordered array of
    size $n$.

-   $\Theta(n)$: Maximum finding in an unordered $n$ element sequence,
    computing the sum of the elements in an array (reduction), computing
    all prefix sums over an array.

-   $\Theta(n\log n)$: Comparison-based sorting of an $n$ element array.

-   $\Theta(n^2)$: Matrix-vector multiplication with dense, square
    matrices of order $n$ (inputs of size $\Theta(n^2)$).

-   $O(n^3)$: Dense matrix-matrix multiplication, which we will take as
    the best bound known to us in this lecture (but far from best known,
    see, e.g., [@Strassen69]).

-   $O(n+m)$: Breadth-First Search (BFS) and Depth-First Search (DFS) in
    graphs with $n$ vertices and $m$ edges.

-   $\Theta(m+n)$: Merging two ordered sequences of length $n$ and $m$
    with a constant time comparison function, identifying the connected
    components of undirected graphs with $n$ vertices and $m$ edges.

-   $O(n\log n+m)$: Dijkstra's Single-Source Shortest Problem algorithm
    on real, non-negative weight, directed graphs with $n$ vertices and
    $m$ arcs using a best known priority queue.

Regardless of how time per processor-core is accounted for, the time of
the parallel algorithm [Par]{.sans-serif} when executed on $p$
processor-cores is the time for the last processor-core to finish,
assuming that all cores started at the same time. Note that we here make
a lot of implicit assumptions, "same time" etc., that will not be
discussed further but are worth thinking much more about. The rationale
for this convention is twofold: Our problem is solved when the last
processor has finished (and we know that this is the case), and since
our parallel system is dedicated, it has to be paid for until all
processor-cores are again free for something else.

In Parallel Computing as a practical, experimental endeavor,
[Seq]{.sans-serif} and [Par]{.sans-serif} denote concrete
implementations of the algorithms, and $T_{\mathsf{seq}}(n)$ and
$T^{p}_{\mathsf{par}}(n)$ are measured running times for concrete,
precisely specified inputs of size $O(n)$ on concrete and precisely
specified systems. Designing measuring procedures and selecting inputs
belong to experimental Computer Science and are highly non-trivial
tasks; they will not be treated in great detail in these lectures.
Suffice it to say that time is measured by starting the processor-cores
at the same time as far as this is possible, and accounting for the time
$T^{p}_{\mathsf{par}}(n)$ by the last processor-core to finish. Inputs
may be either single, concrete inputs or a whole larger set of inputs.
Worst-case inputs may be difficult (impossible) to construct and are
often also not interesting, so inputs are rather "typical" instances,
"average-case" instances, randomly generated instances, inputs with
particular structure, etc.(for recent criticism of and alternatives to
worst-case analysis of algorithms, see [@Roughgarden21]). The important
point for now is that inputs and generally the whole experimental set-up
be clearly described, so that claims and observations can be objectively
verified (reproducibility).

### Speed-up

We measure the gain of the parallel algorithm [Par]{.sans-serif} over
the best known or possible sequential algorithm [Seq]{.sans-serif} for
inputs of size $O(n)$ by relating the two running times. Parallel
Computing aims to improve on the best that we can already do with a
single processor-core. This is the fundamental notion of absolute
*speed-up* over a given baseline:

::: definition
**Definition 2.4** (Absolute Speed-up). *The *absolute speed-up* of
parallel algorithm [Par]{.sans-serif} over best known or best possible
sequential algorithm [Seq]{.sans-serif}(solving the same problem) for
input of size $O(n)$ on a $p$ processor-core parallel system is the
ratio of sequential to parallel running time, i.e., $$\begin{aligned}
  \mathrm{SU}_{p}(n) & = & \frac{T_{\mathsf{seq}}(n)}{T^{p}_{\mathsf{par}}(n)} \quad .
\end{aligned}$$*
:::

The notion of speed-up is meaningful in both theoretical (analyzed, in
some model) and practical (measured running times for specific inputs)
settings. Often, speed-up is analyzed by keeping the problem size $n$
fixed and varying the number of processor-cores $p$ (strong scaling, see
later). Sometimes (scaled speed-up, see later) both input size $n$ and
number of processor-cores $p$ are varied. For the definition, it is
assumed that $T^{p}_{\mathsf{par}}(n)$ is meaningful for any number of
processors $p$ (and any problem size $n$), which for concrete algorithms
and implementations is not always the case: Some algorithms assume
$p=2^d$ for some $d$, a power-of-two number of processors, or
$p=d^2, p=d^3$, a square or cubic number of processors, etc.. The
speed-up is well-defined only for the cases for which the algorithms
actually work. For any input size $n$, there is obviously also a maximum
number of processors beyond which the parallel algorithm does not become
faster (or even work), namely when there is not enough computational
work in the input of size $n$ to keep any more processors busy with
anything useful. Beyond this number, speed-up will decrease: Any
additional processors are useless and wasted.

As an example, a parallel algorithm $\textsf{Par}$ with
$T^{p}_{\mathsf{par}}(n)=O(n/p)$ would have an absolute speed-up of
$O(p)$ for a best known sequential algorithm with
$T_{\mathsf{seq}}(n)=O(n)$, assuming that $n\geq p$ ($p$ in $O(n)$ or,
equivalently, $n$ in $\Omega(p)$). If
$T^{p}_{\mathsf{par}}(n)=O(n/\sqrt{p})$ the speed-up would be only
$O(\sqrt{p})$.

A speed-up of $\Theta(p)$, with upper bounding constant of at most one
and $n$ allowed to increase with $p$, is said to be *linear*, and linear
speed-up of $p$ where both bounding constants are indeed close to one is
said to be *perfect* (by measurement, or by analysis of constants).
Perfect speed-up is rare and hardly achievable (sometimes provably not,
an important example is given later in these lecture notes, see
Theorem [2.10](#thm:prefix-tradeoff){reference-type="ref"
reference="thm:prefix-tradeoff"}).

According to the definitions of linear and perfect speed-up, a parallel
algorithm [Par]{.sans-serif} with running time of at most
$T^{p}_{\mathsf{par}}(n)=c(\frac{n}{p}+\log n)$ for some constant $c$
would have perfect speed-up relative to a best possible sequential
algorithm with running time of at most $T_{\mathsf{seq}}(n)=cn$ steps.
We have $$\begin{aligned}
  \mathrm{SU}_{p}(n) & = & \frac{cn}{c(n/p+\log n)} \\ & = & \frac{p}{1+(p\log
    n)/n}
\end{aligned}$$ which is as close to $p$ as desired for $n/\log n>p$:
For any $\varepsilon, \varepsilon>0$, it holds that
$(p\log n/n)<\varepsilon
\Leftrightarrow n/\log n>p/\varepsilon$. If the sequential and parallel
algorithms have different leading constants $c_0$ and $c_1$,
respectively (with $c_0<c_1$), the speed-up is linear with upper
bounding constant $\frac{c_0}{c_1}<1$. In other words, linear speed-up
means that for any number of processors $p$, the parallel running time
multiplied by $p$ differs by a constant factor from the best (possible
or known) sequential running time (the sequential time being lower) for
sufficiently large $n$; perfect speed-up means that this constant is
practically one.

### "Linear speed-up is best possible" {#sec:linearbest}

Linear speed-up is the best that is possible. The argument for this is
that a parallel algorithm running on $p$ dedicated cores can be
*simulated* on a single core in time no worse than $p
T^{p}_{\mathsf{par}}(n)$ time steps by simulating the steps of the $p$
processors one after the other in a round-robin fashion. If the speed-up
would be more than linear, then
$T_{\mathsf{seq}}(n)>pT^{p}_{\mathsf{par}}(n)$, and the simulated
execution would run faster than the best known sequential algorithm for
our problem, which cannot be. Or: in that case, an even better algorithm
would have been constructed! Sometimes, indeed, a new parallel algorithm
can by a clever simulation lead to a better than previously known
sequential algorithm.

For the PRAM model, the simulation argument can be worked out in detail,
for instance, by writing a sequential simulator for programs in our PRAM
pseudo-code: Within each `par`-construct, execute the instructions of
the assigned processors one after the other in a round-robin fashion,
with some care taken to resolve concurrent writing correctly.

Despite this argument, *super-linear speed-up* larger than the number of
processor-cores $p$ is sometimes reported (mostly in practical
settings) [@FaberLubeckWhite86; @HelmboldMcDowell91]. If the reasons for
this are algorithmic, it can only be that the sequential and parallel
algorithms are, on specific inputs, not doing the same amount of work
(see below). Randomized algorithms, where more and different coin tosses
are possibly done by the parallel algorithm than by the sequential
algorithm, can likewise sometimes exhibit super-linear speed-up. But
also deterministic algorithms, like search algorithms, can exhibit this
behavior if the way the search space is divided over the parallel
processors depends on the number of processor-cores causing the parallel
algorithm to complete the search more than proportionally faster than
the sequential algorithm. Finally, on "real" parallel computing systems,
the memory system and in particular the average memory access times can
differ between algorithms running on a single processor-core and on many
processor-cores where memory is accessed in a distributed fashion and
faster memory "closer to the core" can be used to a larger extent (see
Section [3.1.1](#sec:cachelocality){reference-type="ref"
reference="sec:cachelocality"}).

The argument that linear speed-up is best possible also tells us that
for any parallel algorithm it holds that $T^{p}_{\mathsf{par}}(n) \geq
\frac{T_{\mathsf{seq}}(n)}{p}$. In other words, the best possible
parallel algorithm [Par]{.sans-serif} for the problem solved by
[Seq]{.sans-serif} cannot run faster than $T_{\mathsf{seq}}(n)/p$. This
observation provides us with a first, useful *lower bound* on parallel
running time.

For any parallel algorithm [Par]{.sans-serif} on concrete input of size
$O(n)$, there is, of course a limit on the number of processor-cores
that can be sensibly employed. For instance, putting in more
processor-cores than there is actual work (operations) to be done makes
no sense, and some processors would sit idle for parts of the
computation. Specific speed-up claims are therefore (or should be)
qualified with the range of processor-cores for which they apply.

### Cost and Work {#sec:cost-work}

Our dedicated parallel system with $p$ processor-cores running
[Par]{.sans-serif} is kept occupied for $T^{p}_{\mathsf{par}}(n)$ units
of time, and this is what we have to "pay" for. The *cost* of a parallel
algorithm is, accordingly, defined as the product
$p\times T^{p}_{\mathsf{par}}(n)$. If we picture a parallel computation
as a rectangle with the processor-cores $i$ on one axis, listed densely
from $0$ to $p-1$ and the time spent by the processor-cores on the other
axis, the parallel time $T^{p}_{\mathsf{par}}(n)$ is the largest time
for some processor-core $i$, and the cost is the area of the rectangle
$p\times T^{p}_{\mathsf{par}}(n)$. The parallel algorithm
[Par]{.sans-serif} exploits the parallel system well if the parallel
cost invested for a given input is proportional to the cost of solving
the given problem sequentially by [Seq]{.sans-serif}. This motivates the
notion of *cost-optimality*.

::: {#def:costoptimality .definition}
**Definition 2.5** (Cost-optimal Parallel Algorithm). *A parallel
algorithm [Par]{.sans-serif} for a given problem is *cost-optimal* if
its cost $p
  T^{p}_{\mathsf{par}}(n)$ is in $O(T_{\mathsf{seq}}(n))$ for a best
known sequential algorithm [Seq]{.sans-serif} for any number of
processors $p$ up to some bound that is an increasing function of $n$.*
:::

Cost-optimality requires that, for any given input size $n$, there is a
certain number of processors $p$ for which the cost
$p'T^{p'}_{\mathsf{par}}(n)$ for any $p'\leq p$ is in
$O(T_{\mathsf{seq}}(n))$ and the bounding constant in
$O(T_{\mathsf{seq}}(n))$ does not depend on $p'$ or $p$. The bound on
the number of processors must be an increasing function of the problem
size $n$. The intention is that the cost of [Par]{.sans-serif} is in the
ballpark of the sequential running time of [Seq]{.sans-serif}. Almost
per definition, cost-optimal algorithms have linear speed-up, since
$p T^{p}_{\mathsf{par}}(n)\leq cT_{\mathsf{seq}}(n))$ implies
$\frac{T_{\mathsf{seq}}(n)}{T^{p}_{\mathsf{par}}(n)}\geq \frac{p}{c}$
which is the speed-up. The requirement that the upper bound on the
number of processors $p$ increases with $n$ makes it possible to find an
increasing function of $p$ for which the speed-up is in $\Theta(p)$.
Cost-optimality is a strong property.

A different way of looking at cost-optimality is via the parallel time
complexity and the number of processors needed to reach this fastest
time. The product of this number of processors and this fastest possible
time should still be in the order of the effort required by a best
(known or possible) sequential algorithm. This is captured in the
following definition.

::: {#def:costoptimality2 .definition}
**Definition 2.6** (Asymptotically cost-optimal Parallel Algorithm).
*Let for some given problem [Par]{.sans-serif} be a parallel algorithm
with parallel time complexity $T\infty(n)$. Let $P(n)$ be the smallest
number of processors needed to reach $T\infty(n)$. The cost of
[Par]{.sans-serif} with this number of processors is $P(n)T\infty(n)$
and [Par]{.sans-serif} is cost-optimal if $P(n)T\infty(n)$ is in
$O(T_{\mathsf{seq}}(n))$ for a best known sequential algorithm
[Seq]{.sans-serif} for the given problem.*
:::

We often use the term *work* to quantify the real "effort" that an
algorithm puts into solving one of our computational problems. The work
of a sequential algorithm [Seq]{.sans-serif} on input of size $O(n)$ is
the number of operations (of some kind) carried out by the algorithm.
Sequentially speaking, "work is time". The work of a parallel algorithm
[Par]{.sans-serif} on a system with $p$ processor-cores is the total
work carried out by all of the $p$ cores, excluding time and operations
spent idling by some processors or by processors that are not assigned
to do anything (useful). That is, anything that the cores might be doing
that is not strictly related to the algorithm does not count as work.
With a formal model like the PRAM, this can be given a precise
definition ("work is the operations carried out by assigned
processors"). In more realistic settings, we have to be careful which
idle times should count and which not. The work of parallel algorithm
[Par]{.sans-serif} on input $n$ is denoted $W^{p}_{\mathsf{par}}(n)$.
Ideally, work is independent of the number of processors $p$ and we
might write just $W^{}_{\mathsf{par}}(n)$. This means that the work to
be done by the algorithm [Par]{.sans-serif} has been separated from how
the $p$ processors that will eventually perform this work share the
work. This is a very useful point of view which leads to a productive
separation of concerns between what has to be done ("the work") and who
does it ("which processors"). This point of view motivates the next
definition.

::: definition
**Definition 2.7** (Work-optimal Parallel Algorithm). *A parallel
algorithm [Par]{.sans-serif} with work $W^{}_{\mathsf{par}}(n)$ is
*work-optimal* if $W^{}_{\mathsf{par}}(n)$ is $O(T_{\mathsf{seq}}(n))$
for a best known sequential algorithm [Seq]{.sans-serif}.*
:::

If an algorithm is work-optimal algorithms but not cost-optimal this
indicates either that the way the processors are used in the parallel
algorithms is not efficient (some processors sit idle for too long) or
that most of the work must necessarily be done sequentially, one piece
after the other (because of sequential dependencies). From a
work-optimal algorithm that is not cost-optimal for the first reason, a
better, cost-optimal algorithm with the same amount of work that runs on
fewer processor-cores can sometimes be constructed, but this may not be
easy.

A cost-optimal parallel algorithm is per definition work-optimal but not
the other way around: A parallel algorithm that is not work-optimal
cannot be cost-optimal. Thus, a first step towards designing a good
parallel algorithm is to look for a solution that is (at least)
work-optimal.

Another useful observation following from the notion of parallel work is
that the best possible parallel running time of an algorithm with work
$W^{}_{\mathsf{par}}(n)$ is at least $$\begin{aligned}
T^{p}_{\mathsf{par}}(n) & \geq & \frac{W^{}_{\mathsf{par}}(n)}{p} \quad .
\end{aligned}$$ This is another useful lower bound which is sometimes
called the *Work Law* (See
Section [2.3.1](#sec:taskgraphs){reference-type="ref"
reference="sec:taskgraphs"}). The lower bound is met if the work
$W^{}_{\mathsf{par}}(n)$ that has to be done has been perfectly
distributed over the $p$ processors and no extra costs have been
incurred.

As an extreme example, consider a "parallel" algorithm that is just a
(best) sequential algorithm executed on one out of the $p$ processors.
This is a work-optimal parallel algorithm, but it is clearly not
cost-optimal since all but one processor are idle. Its cost
$O(pT_{\mathsf{seq}}(n))$ is optimal when running it on one or a small,
constant number of processors $p$; but as long as the number of
processors that can be efficiently exploited cannot be increased with
increasing problem size, such an algorithm is not cost-optimal according
to our definition, and speed-up beyond a limited, constant number of
processors cannot be achieved. This is not what is desired of a good
parallel algorithm. Cost- and work-optimality are asymptotic notions of
properties that hold for large problems and large numbers of processors.

Algorithms that are not cost-optimal do not have linear speed-up. The
PRAM maximum finding algorithm of
Theorem [2.1](#alg:fastmax){reference-type="ref"
reference="alg:fastmax"} takes $O(1)$ time with $O(n^2)$ processors and
therefore has cost $O(n^2)$, which is far from
$T_{\mathsf{seq}}(n) = O(n)$. To determine the speed-up of this
algorithm, we first have to observe that the algorithm can be simulated
with $p\leq n^2$ processors in $O(n^2/p)$ parallel time steps. The
speed-up is $\mathrm{SU}_{p}(n)=O(n/(n^2/p))= p/n$. The speed-up is
*not* independent of $n$, and actually decreases with $n$: The larger
the input, the lower the speed-up.

The point of distinguishing work and cost is to separate the discovery
of parallelism from an all too specific assignment of the work to the
actually available processors. A good, parallel algorithm is
work-optimal and can become fast when enough processors are given. A
next design step is then to carefully assign the work to only as many
processors as allowed to keep the algorithm cost-optimal. The PRAM
abstraction supports this strategy well: Processors can be assigned
freely (with the **par**-construct), and the analysis can focus on the
number of operations actually done by the assigned processors (the
work).

More precisely, let us assume that a work-optimal PRAM algorithm with
work $W^{}_{\mathsf{par}}(n)$ and parallel time complexity of
$T\infty(n)$ has been found. Such an algorithm can (in principle) be
implemented to run on a $p$-processor PRAM (same variant) in at most
$\lfloor \frac{W^{}_{\mathsf{par}}(n)}{p}\rfloor+T\infty(n)$ parallel
time steps. This follows easily. In each of the $T\infty(n)$ parallel
steps some amount of work $W^{i}_{\mathsf{par}}(n)$ has to be done. This
work can be done in parallel on the $p$ processors in
$\lceil \frac{W^{i}_{\mathsf{par}}(n)}{p}\rceil$ time steps by a
straightforward round-robin execution of the work units over the $p$
processors. Summing over the steps gives $$\begin{aligned}
  \sum_{i=0}^{T\infty(n)-1} \lceil \frac{W^{i}_{\mathsf{par}}(n)}{p}\rceil
  & \leq & \sum_{i=0}^{T\infty(n)-1} (\lfloor \frac{W^{i}_{\mathsf{par}}(n)}{p}\rfloor+1) \\
  & \leq & \lfloor \frac{W^{}_{\mathsf{par}}(n)}{p}\rfloor+T\infty(n)
\end{aligned}$$

This observation is also known as *Brent's Theorem* [@Brent74]. The
observation only tells us that an efficient execution of the algorithm
is possible on a $p$-processor PRAM, but not how the work units for each
step can be identified. Sometimes this is obvious and sometimes not.

### Relative Speed-up and Scalability

While the absolute speed-up measures how well a parallel algorithm can
improve over its best known sequential counterpart, it does not measure
whether the parallel algorithm by itself is able to exploit the $p$
processors well. This notion of *scalability* is the *relative
speed-up*.

::: definition
**Definition 2.8** (Relative Speed-up). *The *relative speed-up* of a
parallel algorithm [Par]{.sans-serif} is the ratio of the parallel
running time with one processor-core to the parallel running time with
$p$ processor-cores, i.e., $$\begin{aligned}
  \mathrm{SUR}_{p}(n) & = & \frac{T^{1}_{\mathsf{par}}(n)}{T^{p}_{\mathsf{par}}(n)} \quad .
\end{aligned}$$*
:::

Assume that an arbitrary number of processors is available. Any parallel
algorithm has, for any (fixed) input of size $O(n)$, a fastest running
time that it can achieve, denoted by $T\infty(n)$ which is the time
$T^{p'}_{\mathsf{par}}(n)$ for some number of processors $p'$; this was
defined as the parallel time complexity (see
Section [2.2.1](#sec:timecomplexity){reference-type="ref"
reference="sec:timecomplexity"}). Per definition,
$T^{p}_{\mathsf{par}}(n)\geq T\infty(n)$ for any number of processors
$p$. It thus holds that
$\mathrm{SUR}_{p}(n) = \frac{T^{1}_{\mathsf{par}}(n)}{T^{p}_{\mathsf{par}}(n)} \leq
\frac{T^{1}_{\mathsf{par}}(n)}{T\infty(n)}$.

The ratio $\frac{T^{1}_{\mathsf{par}}(n)}{T\infty(n)}$ which is a
function of the input size $n$ only is called the *parallelism* of the
parallel algorithm. It is clearly both the largest, relative speed-up
that can be achieved, as well as an upper bound on the number of
processors up to which linear, relative speed-up can be achieved. If
some number of processors $p'$ larger than the parallelism is chosen,
the definition says that $\mathrm{SUR}_{p'}(n) < p'$, that is, less than
linear speed-up. The parallelism is also the asymptotically smallest
number of processor needed to achieve the best possible running time
$T\infty(n)$.

It is important to clearly distinguish between absolute and relative
speed-up. Relative speed-up compares a parallel algorithm or
implementation against itself, and expresses to what extent the
processors are exploited well (linear, relative speed-up). Absolute
speed-up compares the parallel algorithm against a (best known or
possible) baseline, and expresses how well it improves over the
baseline. A parallel algorithm may have excellent relative speed-up, but
poor absolute speed-up. Is such a good algorithm? In any case, reporting
only the relative speed-up for a parallel algorithm or implementation
can be grossly misleading and should never be done in serious Parallel
Computing. An absolute baseline always must be defined (that which we
want to improve over) and absolute running times also stated. There are
plenty of examples of basing claims on relative speed-ups only also in
the scientific literature. For more on such pitfalls and
misrepresentations, see the now well-known and often paraphrased
"...Ways to fool the masses..." [@Bailey92], see
also <https://blogs.fau.de/hager/archives/5299>.

The absolute speed-up compares the running time of the parallel
algorithm against the running time of a best known or possible
sequential algorithm. For such an algorithm it holds that
$T_{\mathsf{seq}}(n)\leq T^{1}_{\mathsf{par}}(n)$ and therefore
$$\begin{aligned}
  \mathrm{SU}_{p}(n) & \leq & \mathrm{SUR}_{p}(n) \quad .
\end{aligned}$$ The absolute speed-up is at most as large as the
relative speed-up and also in that sense a tougher measure.

### Overhead and Load Balance

A parallel algorithm for a computational problem usually performs more
work than a corresponding best known sequential algorithm. In summary,
such work is termed *overhead*; thus, overhead is work incurred by the
parallel algorithm that does not have to be done by the sequential
algorithm. Beware that this definition tacitly assumes that sequential
and parallel algorithms are somehow similar and can be compared ("extra
work"). This is not always the case. Sometimes, a parallel algorithm is
totally different from the best known sequential algorithm. Overheads
can be caused by several factors, e.g.,

-   preparation of data for other processor-cores,

-   communication between and coordination of processor-cores,

-   synchronization, and

-   algorithmic overheads: extra or redundant work

when compared to a corresponding, somehow similar sequential algorithm.
When a parallel algorithm [Par]{.sans-serif} is derived from a
sequential algorithm [Seq]{.sans-serif}, we can loosely speak of
*parallelization* and say that [Seq]{.sans-serif} has been
*parallelized* into [Par]{.sans-serif}. Parallel algorithms implemented
with [OpenMP]{.roman} (see
Section [3.3](#sec:openmpframework){reference-type="ref"
reference="sec:openmpframework"}) are, for instance, often very concrete
parallelizations of corresponding sequential algorithms. Again, it is
important to stress that many parallel algorithms are specifically not
parallelizations of some sequential algorithm.

Overheads are more or less inevitable, but if they are on the order of
(within the bounds of) the sequential work, $O(T_{\mathsf{seq}}(n))$ the
parallel algorithm can still be work- and cost-optimal, and thus have
linear, although not perhaps perfect speed-up. Often, overheads increase
with the number of processors $p$, giving, for fixed problem size $n$, a
limit on the number of processors that can be used while still giving
linear speed-up. If the overheads are asymptotically larger than the
sequential work, the parallel algorithm will never have linear speed-up.

The overheads caused by communication and synchronization between
processor-cores are often significant. Later in these lecture notes, we
will introduce a simple model for accounting for communication
operations. Suffice it here to say that a simple synchronization between
$p$ processors, which means ascertaining that a processor cannot
continue beyond a certain point in its computation before all other
processors have reached a certain point in their computations (see
Section [2.3.14](#sec:barrier){reference-type="ref"
reference="sec:barrier"}), may (and must) take $\Omega(\log
p)$ operations. An exchange of data will typically take time
proportional to the amount of the data (per processor) and an additive
term dependent on the number of processors $p$.

Between communication operations, the processor-cores operate
independently on parts of the problem although they could interfere
indirectly through the memory and cache system (this will be discussed
in later parts of these lecture notes, see
Section [3.1.1](#sec:cachelocality){reference-type="ref"
reference="sec:cachelocality"}). The length of the intervals between
communication and synchronization operations is sometimes referred to as
the *granularity* of the parallel algorithm. A parallel computation in
which communication and synchronization occur rarely is called *coarse
grained*. If communication and synchronization occur frequently, the
computation is called *fine grained*. These are relative (and vague)
terms. Machine models that can support fine grained algorithms, are also
called fine grained. The PRAM is an extreme example: The processors can
(and often do) communicate via the shared memory in every step, and they
are lock-step synchronized with no overhead for synchronization.

In some parallel algorithms, the processors may not perform the same
amount of work, and/or have different amounts of overhead. If we, for
the moment, let $T^{i}_{\mathsf{par}}(n)$ denote the time taken by some
processor-core $i,0\leq i<p$ from the time this processor-core starts
until it terminates, the (absolute) *load imbalance* is defined as
$$\max_{0\leq i,j<p}{|T^{i}_{\mathsf{par}}(n)-T^{j}_{\mathsf{par}}(n)|}=
  \max_{0\leq i<p}T^{i}_{\mathsf{par}}(n)-\min_{0\leq i<p}T^{i}_{\mathsf{par}}(n) \quad .$$
The relative load imbalance is the ratio of absolute load balance to
parallel time (completion time of slowest processor). Too large load
imbalance is another reason that a parallel algorithm may have a too
small (or non-linear) speed-up. Too large load imbalance may likewise be
a reason why an otherwise work-optimal parallel algorithm is not
cost-optimal: Too many processors take too small a share of the total
work.

Good load balance means that
$T^{i}_{\mathsf{par}}(n)\approx T^{j}_{\mathsf{par}}(n)$ for all pairs
of processors $(i,j)$. Achieving good, even load balance over the
processors is called *load balancing* and is always an issue in
designing a parallel algorithm, explicitly by the construction of the
algorithm or implicitly by taking steps later to ensure a good load
balance. We distinguish between *static load-balancing*, where the
amount of work to be done can be divided upfront among the processors,
and *dynamic load balancing*, where the processors have to communicate
and exchange work during the execution of the parallel algorithm. Static
load balancing can be further subdivided into *oblivious, static
load-balancing*, where the problem can be divided over the processors
based on the input size and structure alone but regardless of the actual
input, and *adaptive, problem-dependent, static load-balancing*, where
the input itself is needed in order to divide the work and preprocessing
may be required. Some aspects of the load balancing problem
(work-stealing, loop scheduling) will be discussed later in this part of
the lecture notes. However, load balancing *per se* is too large a
subfield of Parallel Computing to be treated in much detail here.

Problems and algorithms where the input and work can be statically
distributed to the processors and where no further explicit interaction
is required are called either *embarrassingly parallel*, *trivially
parallel*, or *pleasantly parallel*. These are the best (but
uninteresting, in the sense of being unchallenging) cases of easily
parallelizable problems with linear or even perfect speed-up. The
realization that the problem is trivially or embarrassingly parallel
can, of course, be highly non-trivial and the way to see this
unpleasant.

### Amdahl's Law

Gene Amdahl made a simple observation on how to speed up
programs [@Amdahl67], which when applied to Parallel Computing yields
severe bounds on the speed-up that certain parallel algorithms can
achieve. The observation assumes that the parallel algorithm is somehow
derived by parallelization of the sequential algorithms.

::: theorem
**Theorem 2.4** (Amdahl's Law). *Assume that the work performed by
sequential algorithm [Seq]{.sans-serif} can be divided into a strictly
sequential fraction $s, 0<s\leq 1$, independent of $n$, that cannot be
parallelized at all, and a fraction $r=(1-s)$ that can be perfectly
parallelized. The parallelized algorithm is [Par]{.sans-serif}. Then,
the maximum speed-up that can be achieved by [Par]{.sans-serif} over
[Seq]{.sans-serif} is bounded by $1/s$.*
:::

The proof is straightforward. With the assumption that $$\begin{aligned}
  T^{p}_{\mathsf{par}}(n) & = & sT_{\mathsf{seq}}(n)+\frac{(1-s)T_{\mathsf{seq}}(n)}{p}
\end{aligned}$$ we get $$\begin{aligned}
  \mathrm{SU}_{p}(n) & = & \frac{T_{\mathsf{seq}}(n)}{sT_{\mathsf{seq}}(n)+\frac{(1-s)T_{\mathsf{seq}}(n)}{p}} \\
  & = & \frac{1}{s+\frac{1-s}{p}} \\
  & \rightarrow & \frac{1}{s}\ \mbox{for $p\rightarrow\infty$} \quad .
\end{aligned}$$

Amdahl's Law is devastating. Even the smallest, constant sequential
fraction of the algorithm to be parallelized will limit and eventually
kill speed-up. A sequential fraction of $10$%, or $1$%, sounds
reasonable and harmless but limits the speed-up to $10$, or $100$, no
matter what else is done, no matter how large the problem, and no matter
how many processors are invested. Note that the parallelization
considered is work-optimal; but it is surely not cost-optimal. The
running time of the parallel algorithm is at least
$sT_{\mathsf{seq}}(n)$ and since $s, s<1$ is constant, the cost is
therefore $O(pT_{\mathsf{seq}}(n))$ which is not in
$O(T_{\mathsf{seq}}(n))$.

A sequential algorithm which falls under Amdahl's Law cannot be used as
the basis of a good, parallel algorithm: Its speed-up will be severely
limited and bounded by a constant. Amdahl's Law is therefore rather an
analysis tool: If it turns out that a (large) fraction of the algorithm
at hand cannot be parallelized, we have to look for a different, better
algorithm. This is what makes Parallel Computing a creative activity:
Simple parallelization of a sequential algorithm will often not lead to
a good, parallel counterpart. New ideas for old problems are sometimes
needed.

Typical victims of Amdahl's Law are:

-   Input/output: For linear work algorithms, reading the input and
    possibly also writing the output will take $\Omega(n)$ time steps,
    and thus be a constant fraction of $O(n)$.

-   Sequential preprocessing: As above.

-   Maintaining sequential data structures, in particular sequential
    initialization, can easily turn out to be a constant fraction of the
    total work.

-   Hard-to-parallelize parts that are done sequentially (which might
    look innocent enough for just small parts): If such parts take a
    constant fraction of the total work, Amdahl's Law applies.

-   Long chains of dependent operations (operations that have to be
    performed one after the other and cannot be done in parallel), not
    necessarily on the same processor-core.

When analyzing and benchmarking parallel algorithms, input/output is
often disregarded when accounting for sequential and parallel time. The
defensible reason for this is that we are interested in how the core
parallel algorithm performs (speeds up), under the assumption that the
input has already been read and properly distributed to the
processor-cores according to the specification. In these lecture notes,
our algorithms are small parts (building blocks) of larger applications
and in this larger context would not need input/output: The data are
already where they should be. Also results do not have to be output but
should just stay and be available for the next building block to use.
We, therefore, analyze the building blocks in isolation without the
input/output parts that might fall victim to Amdahl's Law.

In a good parallel algorithm, not falling victim to Amdahl's Law, the
sequential part $s(n)$ will not be a constant fraction of the total work
but depend on and decrease with $n$. If such is the case, Amdahl's Law
does not apply. Instead, a good speed-up can be achieved with large
enough inputs. Parallel Computing is about solving large, work-intensive
problems, and in good parallel algorithms the parts doing the parallel
work dominate the total work as the input gets large enough.

### Efficiency and Weak Scaling

As observed, there is, for any parallel algorithm on input of size
$O(n)$, always a fastest possible time, $T\infty(n)$, that the algorithm
can achieve (the parallel time complexity). Thus, the parallel running
time of an algorithm with good, linear speed-up (up to the number of
processor-cores determined by the parallelism), can be written as
$T^{p}_{\mathsf{par}}(n)=O(T(n)/p+t(n))$, that is, as a parallelizable
term $T(n)$ and a non-parallelizable term $t(n)=T\infty(n)$. If speed-up
is not linear, the parallel running time is instead something like
$T^{p}_{\mathsf{par}}(n)= O(T(n)/f(p)+t(n))$ strictly with $f(p)<p$ and
$f(p)$ in $o(p)$, or $T(n)$ is not in $O(T_{\mathsf{seq}}(n))$.

If we compare against a sequential algorithm with
$T_{\mathsf{seq}}(n)=O(T(n))=
O(T(n)+t(n))$, a parallel algorithm where $t(n)/T(n)
\rightarrow 0$ as $n\rightarrow\infty$ is also good and can have linear
speed-up for large enough $n$. The speed-up is namely
$$\mathrm{SU}_{p}(n) = \frac{T_{\mathsf{seq}}(n)}{T^{p}_{\mathsf{par}}(n)} =
  O(\frac{T(n)}{T(n)/p+t(n)}) = O(\frac{1}{1/p+t(n)/T(n)}) \rightarrow
  O(p)$$ as $n$ increases. This is called *scaled speed-up*, and the
faster $t(n)/T(n)$ converges, the faster the speed-up becomes linear.
Against Amdahl's Law, the sequential part $t(n)$ should be as small as
possible and increase more slowly with $n$ than the parallelizable part
$T(n)$. Algorithms with this property are cost-optimal according to
Definition [2.5](#def:costoptimality){reference-type="ref"
reference="def:costoptimality"}.

It is a good way which we use throughout these lecture notes to state
the performance of a (work-optimal) parallel algorithm as
$T^{p}_{\mathsf{par}}(n)=O(T(n)/p+t(n,p))$ with the assumption that
$t(n,p)$ is in $O(T(n))$ for fixed $p$, and
$T_{\mathsf{seq}}(n)=O(T(n))$. That is, we allow the non-parallelizable
part to depend on both $n$ and $p$. Often, however, $t(n,p)$ is just
$t(n)$ independent of $p$ or $t(p)$ depending on $p$ only
(synchronization costs). An iterative parallel algorithm with a
convergence check involving synchronization could, for instance, run in
$O(n/p+\log
n\log p)$ parallel time with $t(n,p)=O(\log n\log p)$. Such an algorithm
would perform total linear $O(n)$ work which has been well distributed
over the $p$ processors; the algorithm performs $O(\log
n)$ iterations each of which incurs a synchronization overhead of
$O(\log p)$ operations.

The *parallel efficiency* of a parallel algorithm [Par]{.sans-serif} is
measured by comparing [Par]{.sans-serif} against a best possible
parallelization of [Seq]{.sans-serif} as given by the Work Law (see
Section [2.3.1](#sec:taskgraphs){reference-type="ref"
reference="sec:taskgraphs"}).

::: definition
**Definition 2.9** (Parallel Efficiency). *The efficiency
$\mathrm{E}_{p}(n)$ for input of size $O(n)$ and $p$ processors of
parallel algorithm [Par]{.sans-serif} compared to sequential algorithm
[Seq]{.sans-serif} is defined as
$$\mathrm{E}_{p}(n) = \frac{T_{\mathsf{seq}}(n)}{p} \big/ T^{p}_{\mathsf{par}}(n) =
  \frac{T_{\mathsf{seq}}(n)}{p T^{p}_{\mathsf{par}}(n)} = \frac{\mathrm{SU}_{p}(n)}{p} \quad .$$*
:::

As worked out in the definition, the efficiency is also the achieved
speed-up divided by $p$ as well as the sequential time divided by the
cost of the parallel algorithm. It therefore holds that

-   $\mathrm{E}_{p}(n)\leq 1$.

-   If $\mathrm{E}_{p}(n)=e$ for some constant $e, 0<e\leq 1$, the
    speed-up is linear.

-   Cost-optimal algorithms have constant efficiency.

Should it happen that the efficiency $\mathrm{E}_{p}(n)$, contrary to
the statement above, for some $n$ and number of processors $p$ is larger
than $1$, equivalently that the absolute speed-up is larger than $p$,
this tells us that the sequential baseline is not the best (known)
possible. It can be replaced by some variation of the parallel
algorithm. In such a case, Parallel Computing has helped to discover a
better sequential algorithm for the given problem.

We note that this is a definition of *algorithmic efficiency*: How close
is the time of the parallel algorithm with $p$ processors to that of a
best possible parallelization of a best (known) sequential algorithm?
This definition does not say anything about how well the parallel or
sequential algorithm exploits the hardware capabilities and how close
the performance can come to the nominal performance of the parallel
processor system at hand. This notion of *hardware efficiency* plays a
role in High-Performance Computing (HPC), understood here as the
discipline of getting the best out of the given system.

If an algorithm does not have constant efficiency and linear speed-up
for fixed, constant input sizes $n$, we can try to maintain a desired,
constant $e$ efficiency by instead increasing the problem size $n$ with
the number of processors $p$. This is the notion of
*iso-efficiency*[@GramaGuptaKumar93; @GramaKarypisKumarGupta03] and can
be achieved for cost-optimal algorithms.

::: definition
**Definition 2.10** (Weak Scalability (constant efficiency)). *A
parallel algorithm [Par]{.sans-serif} is said to be *weakly scaling*
relative to sequential algorithm [Seq]{.sans-serif} if, for a desired,
constant efficiency $e$, there is a slowly growing function $f(p)$ such
that the efficiency is $\mathrm{E}_{p}(n)=e$ for $n$ in $\Omega(f(p))$.
The function $f(p)$ is called the *iso-efficiency function*.*
:::

How slowly should $f(p)$ grow? A possible answer is found in another
definition of weak scaling.

::: definition
**Definition 2.11** (Weak Scalability (constant work)). *A parallel
algorithm [Par]{.sans-serif} with work $W^{}_{\mathsf{par}}(n)$ is said
to be *weakly scaling* relative to sequential algorithm
[Seq]{.sans-serif} if, by keeping the average work per processor
$W^{}_{\mathsf{par}}(n)/p$ constant at $w$, the running time of the
parallel algorithm $T^{p}_{\mathsf{par}}(n)$ remains constant. The input
size scaling function is $g(p) = T_{\mathsf{seq}}^{-1}(pw)$.*
:::

Ideally, the iso-efficiency function $f(p)$, which tells how $n$ should
grow as a function of $p$ to maintain constant efficiency, should not
grow faster than the input size scaling function $g(p)$, which tells how
much $n$ can at most grow if the average work is to be kept constant:
$f(p)$ should be $O(g(p))$. The two notions may contradict. Constant
efficiency could require larger $n$ than permitted for maintaining
constant average work. This happens if the sequential running time is
more than linear. Keeping constant efficiency requires $n$ to increase
faster than allowed by constant work weak scaling. For such algorithms,
constant work is maintained with decreasing efficiency.

### Scalability Analysis

How well does a parallel algorithm or implementation now perform against
a sequential counterpart for the problem that we are interested in, in
particular how well can it exploit the available processor resources?
*Scalability analysis* examines this, theoretically and practically by
analyzing (measuring) the parallel time that can be reached for
different number of processors $p$ and possibly different problem sizes
$n$.

-   Strong scaling analysis: Keep the input (size) $n$ constant. The
    algorithm is *strongly scalable* up to some maximum number of
    processors, as expressed by the parallelism of the algorithm if the
    parallel time decreases proportionally to $p$ (linear speed-up).

-   Weak scaling analysis: Keep the average work per processor constant
    by increasing $n$ with the number of processors $p$. The algorithm
    is *weakly scalable* if the parallel running time remains constant
    with increasing number of processors.

A strongly scaling algorithm, a strong property, is able to speed up the
solution of the given problem for some fixed size $n$ (large enough for
parallel execution to make sense) proportionally to the number of
employed processor-cores: our primary Parallel Computing goal. A weakly
scaling algorithm in the sense of constant work per processor is able to
solve larger and larger instances of the problem within an allotted time
frame. Ideally, the time spent when the processor-cores are performing
the same amount of work remains constant regardless of the number of
processors employed. If this is not the case, and the parallel time is
increasing with the number of processors, this indicates that the
parallelization overhead (due to communication, synchronization,
unfavorable load balancing, or redundant computation) is increasing with
$p$.

### Relativized Speed-up and Efficiency

For very large parallel systems with tens or hundred thousands of
processor-cores, measuring speed-up relative to a sequential baseline
running on one processor may not make sense or even be possible. The
problem size needed to keep the extreme number of processors busy may
simply be too large (and time consuming) to run on a single processor.
Scalability analysis may in such cases use as baseline the parallel
algorithm running on some number $p'$ of processors (say, $p'=2, p'=100,
p'=1000$ processor-cores). What happens if the number of processors is
doubled? What happens when going from $p'$ to $2p'$ to $10p'$ or to some
$p>p'$ processors? Does the problem size need to increase to maintain a
certain efficiency? The definitions of relative speed-up and (relative)
efficiency can easily be modified to use a different processor baseline
$p'$.

### Measuring parallel Time and Speed-up empirically

Running parallel programs on a parallel multi-core processor or a large
parallel computing system requires quite considerable support from the
system's run-time system: Processor-cores must be allocated to the
program, the program's active entities (processes, threads, ...) must be
started and so on, the execution monitored, the program execution
terminated, and the resources be given free for the next program to use.
The measured time for running a full, parallel application is taken as
the *wall-clock time* from starting the application until the system is
free again, in accordance with our definition of parallel time and
assuming that accurate timers are available, and therefore includes all
these surrounding "overheads". Benchmarking and assessing the
performance ("is this good enough?") of an application in this context
is done by varying the inputs, the number of processors used, the
system, and other relevant factors in a systematic and well-documented
way.

Parallel Computing is most often concerned with the algorithmic building
blocks of such larger applications and these building blocks are the
computational problems we are studying. Benchmarking and performance
assessment is therefore rather done by conducting dedicated experiments,
possibly using specific benchmarking tools, with our developed kernels
and building blocks. A benchmarking program or tool will invoke the
kernel to be benchmarked in a controlled manner. For Parallel Computing
with our definition of parallel time, it is thus common to assume (and
therefore ensure) that the available processor-cores to be used in the
assessment will start at the same time (as far as this makes sense).
This will entail some form of *temporal synchronization* between the
processor-cores, which is in itself a non-trivial problem in Parallel
Computing. Also some means of detecting which processor-core was the
last to finish is needed, possibly by again synchronizing the
processor-cores. As always in experimental science, measurement and
synchronization should be non-intrusive and not affect or distort the
experimental outcome, which is another highly non-trivial issue. Since
computer systems are effectively not deterministic objects (with respect
to timing) and measured run-times may fluctuate from run to run, kernel
benchmarks are repeated a certain number of times, say $10$, $30$, $100$
times, or until results are considered stable enough under some
statistical measure, or until the experimenter runs out of time. The
time reported by the experiment as the parallel running time of the
algorithm in question may be based on a statistical measure like average
time of the slowest processor-core over the repetitions or the median
time of the measured times. Sometimes, the fastest time over the
repetitions of the slowest processor-core in each repetition is taken as
the parallel running time. The argument for this is that this best time
that the system could produce can be reproducible and stable over
repeated experiments. A good experiment will clearly describe the
experimental setup and the statistics used in computing and reporting
the run-times. For others to reproduce an experiment and verify claims
on performance, a precise description of the parallel systems is
likewise required: Processor architecture, instruction set, number of
processor-cores, organization and grouping of the cores, clock
frequency, memory, cache sizes, etc..

In these lecture notes, asymptotic worst-case analysis is used to judge
and compare algorithms, but most often worst-case inputs are not known
and may also not be interesting, common use-cases at all. Experimental
analysis aims at showing performance under many different inputs, in
particular those that are realistic and typical for the uses of the
algorithmic kernel under examination. Experiment design deals with the
construction of good experimental inputs. For non-oblivious algorithms
that are sensitive to the actual input (and not only the size of the
input) it is good practice to always consider extreme and otherwise
special case inputs, such as are expected to lead to either extremely
good or extremely bad performance. Average case and otherwise "typical"
inputs are likewise probably of interest and should be considered.

In Parallel Computing we are most often interested in aspects of
scalability in problem size and in particular in number of
processor-cores. On both accounts, it can be considered bad practice to
focus only on input sizes $n$ and especially number of processors $p$
that are powers of two. The reason for this is that in many algorithms,
powers-of-two are special, and performance in these cases might be
either extremely good or extremely bad. In particular, parallel
algorithms are sometimes designed around communication structures or
patterns where the number of processors is first considered to be some
$p=2^q$. Likewise, some algorithms, for instance, dealing with
two-dimensional matrices, may be special for inputs and number of
processor-cores that are square numbers. Benchmarking for only inputs
$n$ and $p$ that are squares can likewise be highly misleading. Excluded
from these considerations are of course algorithms and kernels that only
work for such special numbers.

### Examples {#sec:scalabilityexamples}

It is illustrative(!) to strengthen intuition to visualize parallel
running time, (absolute) speed-up, efficiency, and iso-efficiency as
functions of the number of processors put into solving a problem of size
$n$ (for different $n$). Let some such problems be given with best known
sequential running times $O(n)\leq c n$, $O(n\log n)\leq c
(n\log n)$, and $O(n^2)\leq c n^2$ as seen many times now in these
lecture notes, for some bounding constant $c, c>0$ (the notation is
sloppy: We mean that the constant of the dominating term hidden within
the $O$ is $c$).

We first assume that the linear $O(n)$ algorithm has been parallelized
by algorithms running work-optimally in $O(n/p+1)\leq C(n/p+1)$,
$O(n/p+\log p)\leq C(n/p+\log p)$, $O(n/p+\log n)\leq C(n/p+\log n)$,
and $O(n/p+p)\leq C(n/p+p)$, respectively, for some bounding constant
$C, C>0$: Also many examples of such algorithms have been (and will be)
seen in the lecture notes.

We first assume that the bounding constants in our sequential and
parallel algorithms are "in the same ballpark", and normalize both
constants to $c=C=1$. We plot the parallel running time as functions of
the number of processors $p$ for $1\leq p\leq 128$, and take
$n=128, 128^2$, respectively; these are really small problems for a
linear time algorithm, $128^2=16K$ (and even $128^3=2M$). The running
times are shown in the following two plots.

::: center
:::

::: center
:::

The running time (number of steps) plots do not very well differentiate
the four different parallel algorithms. For the larger problem size,
$n=128^2$, there is virtually no difference to be seen. The shape of the
curves for these linearly (perfect) scaling algorithms is hyperbolic
(like $1/p$). The parallel algorithm with running time $O(n/p+p)$ is
interesting: For the small input with $n=128$, running time decreases
until about $p=10$ processors, and then increases. Indeed the best
possible running time of this algorithm is $T\infty(n)=\sqrt{n}$, and
the parallelism is also $n/\sqrt{n}=\sqrt{n}$. This can be seen by
minimizing $C(n/p+p)$ for $p$, which can be done by solving $Cn/p=Cp$
for $p$, giving $p=\sqrt{n}$ (or more tediously, by calculus).

Plotting instead the absolute (unit-less) speed-up against the linear
(best known) $O(n)$ algorithm (with $c=C=1$) can highlight the actually
different behavior of the four parallel algorithms. We plot for three
problem sizes $n=128,128^2,128^3$.

::: center
:::

::: center
:::

::: center
:::

Speed-up for the small problem size $n=128$ is not impressive and as we
would like, except for the first parallel algorithm. This changes
drastically and impressively as $n$ grows. Indeed, for the "large"
$n=128^3$ problem, all four parallel algorithms show perfect speed-up of
almost $128$ for $p=128$.

If there is a difference in the bounding constants between sequential
and parallel algorithms, say $c=1$ and $C=10$, which means that the
parallel algorithm is a constant factor of $10$ slower than the
sequential one when executed with only one processor, speed-ups change
proportionally:

::: center
:::

Here, only $1/C$th of the processors are doing productive work in
comparison to the sequential algorithm. Constants *do* matter, and it is
obviously important that sequential and parallel algorithms have leading
constants in the same ballpark. Otherwise, a proportional part of the
processors is somehow wasted.

The parallel efficiency indicates how well the parallel algorithms
behave in comparison to a best possible parallelization with running
time $cn/p$. The (unit-less) parallel efficiencies for the four parallel
algorithms are plotted for $n=128,
128^2, 128^3$.

::: center
:::

::: center
:::

::: center
:::

Indeed, for work-optimal parallelizations, the efficiency improves
greatly with growing problem size $n$ and is already for $n=128^3$ very
close to $1$ for all of the four parallelizations. The iso-efficiency
functions more precisely tell how problem size must increase with $p$ in
order to maintain a given constant efficiency $e$. We calculate the
iso-efficiency functions for the parallel algorithms as follows.

-   For parallel running time $n/p+1$ and desired efficiency $e$, we
    have $e=n/(p(n/p+1))= n/(n+p)\Leftrightarrow e(n+p) =
        n\Leftrightarrow n = ep/(1-e)$.

-   For parallel running time $n/p+\log p$ and desired efficiency $e$,
    we have $e=n/(p(n/p+\log p)) = n/(n+p\log p)\Leftrightarrow
        e(n+p\log p)=n\Leftrightarrow n=ep\log p/(1-e)$

-   For parallel running time $n/p+p$ and desired efficiency $e$, we
    have $e=n/(p(n/p+p))=n/(n+p^2)\Leftrightarrow
        e(n+p^2)=n\Leftrightarrow n=ep^2/(1-e)$

The case with parallel running time $n/p+\log n$ is more difficult. The
efficiency calculation gives $e=n/(p(n/p+\log
n))=n/(n+p\log n)$ and therefore $n/\log n = ep/(1-e)$, for which we do
not know an analytical solution.

We plot the three analytical iso-efficiency functions below for
$p,1\leq p\leq 512$ and $e=90\%$.

::: center
:::

For the first two parallel algorithms, the iso-efficiency function is
indeed "slowly growing", and according to the first definition of weak
scalability, these algorithms are both strongly and weakly scaling. With
the last function, where the iso-efficiency function is in $O(p^2)$, it
is a matter of taste whether to still consider it slowly growing. In the
speed-up plots, we indeed let $n$ grow exponentially
$n=128, 128^2, 128^3$, and the speed-up for the latter algorithms was
excellent.

We now look at non-linear time sequential algorithms. The $O(n\log n)$
algorithm could be a sorting algorithm (mergesort, say) which could have
been parallelized with running time $O((n\log n)/p+\log^2
n)$. The second algorithm is perhaps matrix-vector multiplication, which
can easily be done work-optimally in parallel time $O(n^2/p+n)$ (but
also faster).

The corresponding speed-ups for $n=100,1\,000,10\,000,100\,000$ and
$p,1\leq p\leq 1000$ are shown below.

::: center
:::

::: center
:::

::: center
:::

::: center
:::

The parallelization of the low complexity algorithm with sequential
running time $O(n\log n)$ does not scale as well as the other algorithm.
For an $O(n^2)$ algorithm, an input of size $n=100\,000$ is already
large, and we did not plot for this large $n$ here. However, both
algorithms clearly approach a perfect speed-up with growing $n$.

Finally, we illustrate what happens with non work-optimal parallel
algorithms. Assume we have a parallel algorithm with running times
$O(n\log n/p+1)$ relative to a linear time sequential algorithm, an
$O(n^2/p+n)$ parallel algorithm relative to an $O(n\log n)$ best
possible sequential algorithm, and an Amdahl case where the parallel
algorithm has a sequential fraction $s, 0<s<1$ and parallel running time
$O(sn+(1-s)n/p)$. Lastly, a parallel algorithm with a running time of
$O(n/\sqrt{p}+\sqrt{p})=O((n\sqrt{p})/p+\sqrt{p})$ relative to an
algorithm that solves an $O(n)$ problem.

::: center
:::

::: center
:::

The two plots illustrate the Amdahl case well: Speed-up is bounded by
$1/s$ (here $10$ for $s=10\%$) independently of $n$. The first two
algorithms have a diminishing speed-up with increasing $n$. These two
algorithms have parallel work determined by the problem size which is
asymptotically larger than the sequential work. For the last algorithm,
the parallel work increases "slowly" by a factor of $\sqrt{p}$ with $p$,
and therefore the speed-up of this algorithm does indeed improve with
increasing problem size $n$, but is $o(p)$ and not linear.

## Third block (1-2 Lectures) {#sec:patterns}

In this part of the lecture notes, we take a closer look at the way
(parallel) work may be structured. The most important structures
discussed are work expressed as dependent tasks and work expressed as
loops of independent iterations. The latter can be seen as an expression
of recurring, similar computations in algorithms, pseudo-code and actual
programs: A *parallel programming pattern* or *parallel design pattern*.
The later part of this lecture block gives further examples of parallel
algorithmic design patterns for which (good) parallelizations are known,
including pipeline, stencil, master-slave/master-worker, reductions,
data redistribution, and barrier synchronization. Parallel design
patterns can, explicitly and implicitly, provide useful guidance for
building parallel applications, sometimes even as concrete building
blocks [@MattsonSandersMassingill05; @McCoolRobisonReinders12]. We
illustrate many of the patterns by sequential code snippets using
C [@KernighanRitchie88] to specify the intended outcome and semantics
and use these descriptions to argue for lower bounds on the parallel
performance with given numbers of processors $p$.

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

### pool-workers/Master-slave {#sec:masterworker}

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

