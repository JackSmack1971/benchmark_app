# **🛡️ Architectural Audit: Data Layer Vectorization & PyArrow Memory Layout**

## **1\. Strategic Assessment: The Scalar Allocation Threat**

Your architectural thesis correctly identifies a severe processing bottleneck within the application's data layer. Relying on native Python structures (lists, dictionaries) to accumulate telemetry data introduces massive heap fragmentation and garbage collection overhead. When an application loops over raw lists to perform scalar math (via the standard library statistics module), it fundamentally traps execution within the Global Interpreter Lock (GIL). As multi-request benchmark suites scale to hundreds or thousands of concurrent runs, this scalar iteration pattern monopolizes CPU cycles and unnecessarily bloats the memory footprint, creating a severe processing bottleneck before the UI rendering phase even begins.

To achieve production-grade scalability, the architecture must abandon Python's native heap and push all heavy computations down to C-level vectorized routines utilizing columnar, cache-local memory layouts via Apache Arrow.

## **2\. Technical Validation: Codebase Vulnerability Confirmation**

A surgical audit of the provided benchmark\_app/processing.py source code reveals that **the application currently suffers from the exact architectural anti-pattern you described.** While the codebase attempts to utilize PyArrow, it applies the optimization at the wrong stage of the pipeline, rendering it largely ineffective for the heavy aggregation workloads.

### **A. Ingestion Bottleneck: Native Heap Allocation**

The aggregate\_stats function intercepts the raw benchmark objects and loops through them sequentially, allocating metrics via native .append() operations into an unoptimized dictionary.

**Vulnerability Proof (processing.py, lines 58-69):**

        if result.is\_error:  
            bucket\["errors"\] \+= 1  
        else:  
            bucket\["latencies"\].append(result.latency\_sec)  
            if result.ttft\_sec is not None:  
                bucket\["ttfts"\].append(result.ttft\_sec)  
            bucket\["tps\_vals"\].append(result.tokens\_per\_sec)

*Logic Validation:* This O(N) loop forces Python to dynamically resize arrays and allocate individual float objects on the heap. It fundamentally prevents any vectorized memory optimizations during the data ingestion phase.

### **B. Computational Bottleneck: Scalar Math**

The build\_leaderboard\_rows function extracts these unoptimized lists and iterates through them again, computing metrics using Python's scalar statistics module.

**Vulnerability Proof (processing.py, lines 94-98):**

        avg\_tps \= stats.mean(tps\_vals) if tps\_vals else 0.0  
        std\_tps \= stats.stdev(tps\_vals) if len(tps\_vals) \>= 2 else 0.0  
        cv\_tps \= (std\_tps / avg\_tps \* 100.0) if avg\_tps \> 0 else 0.0

*Logic Validation:* Standard library math functions execute sequentially in Python space. They cannot utilize SIMD (Single Instruction, Multiple Data) CPU instructions, completely wasting the hardware's computational potential.

### **C. The Illusion of PyArrow Compliance**

The module *does* implement PyArrow types, but **only as a post-processing facade**.

**Vulnerability Proof (processing.py, lines 167-184):**

    arrow\_dtypes: dict\[str, str\] \= {  
        "Latency (s)": "float64\[pyarrow\]",  
        "tok/s": "float64\[pyarrow\]",  
        \# ...  
    }  
    for col, dtype in arrow\_dtypes.items():  
        if col in df.columns:  
            df\[col\] \= df\[col\].astype(dtype)

*Logic Validation:* Casting the DataFrame to \[pyarrow\] string representations *after* all the heavy mathematical aggregations have been executed on the scalar lists provides negligible performance benefits. The engine only vectorizes the final 10-row leaderboard output, completely missing the thousands of raw telemetry rows that actually required the optimization.

## **3\. Engineering Execution Conclusion & Refactoring Directive**

The processing.py module is structurally vulnerable to scalar memory bottlenecks and fails the required vectorization standard. To establish a highly reactive, scalable backend, this module requires immediate and surgical refactoring.

**Mandated Architecture Transition:**

1. **Immediate DataFrame Ingestion:** The aggregate\_stats and build\_leaderboard\_rows functions must be collapsed. Raw all\_results dataclasses must be immediately fed into a Pandas DataFrame upon entry: df \= pd.DataFrame(\[r.to\_dict() for r in all\_results\]).  
2. **Strict PyArrow Backend:** The engine must enforce pd.options.mode.dtype\_backend \= "pyarrow" globally, ensuring the initial ingestion bypasses NumPy arrays entirely in favor of zero-copy Arrow memory.  
3. **Vectorized .groupby() Math:** All scalar loops must be eradicated. The leaderboard must be generated utilizing hardware-accelerated groupings: df.groupby('model\_id').agg(avg\_tps=('tokens\_per\_sec', 'mean'), std\_tps=('tokens\_per\_sec', 'std')).  
4. **Method Chaining:** Implement strict .assign() and .pipe() chains to handle downcasting (float32\[pyarrow\], int32\[pyarrow\]) and derived calculations (e.g., CV%).

**Status:** Codebase is non-compliant with the vectorized PyArrow standard. The raw data ingestion loops must be refactored to unlock C-level columnar memory performance and unblock the asynchronous execution thread.