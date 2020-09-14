# QOSF Quantum Computing Mentorship Program - Solution to the Screening Task 1

The present project is an attempt to solve the Screening Task 1 for the admission at the QOSF Quantum Computing Mentorship Program (https://qosf.org/qc_mentorship/; see https://docs.google.com/document/d/1Ow3v8Y4rYBdgxXNxKV9ZUAM4bwL6211U6DWCcByZ4A4/edit for the details of the Screening Tasks).

# Setup information

The project is developed in Python, and makes use of some libraries, including Qiskit (https://qiskit.org/), which is the adopted Quantum-Computing tool. In order to make the project working, you need to perform the following actions:
1) Install Python3.5+ and Anaconda, as described in the first paragraphs of https://qiskit.org/documentation/install.html; 
2) Following the next paragraphs of the above documentation, create a Python Virtual Environment in a folder and activate it;
3) Clone this repository in the newly-created folder (`git clone https://github.com/balordo85/qosf_screening_1`);
4) Run `pip install -r requirements.txt`.
The script is now ready to be run.

# Run information

The script must be run from command line as `python task.py <Lmax> [<yellow_gate> <green_gate>]`. The arguments must follow this convention:
1) At least one argument is required: it is the circuit length `Lmax`;
2) Third and fourth arguments are optional (in case, they have to be both specified). They can take their values in `('x', 'y', 'z')`, and are, respectively, the rotation axes for the parametric gates in the yellow and green blocks. They default to z and x, as asked by the task; freedom for their choice is left in order to answer the bonus question (see below). E.g., executing the script with `y` and `x` as third and fourth argument will place the gates `Ry` and `Rx` in the yellow and green blocks, instead of `Rx` and `Rz`.

# Reports

Each execution of the script, if complete, produces two output files. The name of the files, as it can be seen from the sample outputs attached to the project, report the values of the run arguments, as well as the time stamp at which the files are created. The two files are a `.png` plotting &epsilon; as a function of the circuit size L, and a `.txt` containing the detailed output of each execution.

# Solution details

The code is very simple, and I hope clear; very few explanations should be needed. The script is divided in three parts:
1) Declaration of constants (actually, only the random vector &phi; appearing in the definition of &epsilon;);
2) Declaration of functions: they basically involve the definition of the circuit, the computation of the distance and the generation of the report;
3) Main function: it is where the computation of &epsilon; is triggered. The minimization procedure makes use of the conjugate-gradient descent algorithm, within the implementation provided by the `scipy.optimize.fmin_cg` function (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_cg.html).

# Commenting the results

The main lesson I can draw from the solution of the task is that the proposed circuit can reproduce, when combined with an optimization procedure as a function of &theta;, any 4-qubit state, provided that the circuit size is large enough. Indeed, for L = 4 or 5 or larger, &epsilon; is undistinguishable from 0. The physical reason lying behind this behavior is that the `Rx`-`Ry` transforms the initial `|0>` state for each qubit into a superposition of the two computational basis states, and then the symmetric application of the `Cz` gates produces a highly-entangled 4-qubit state. By moving from this state by gradient-descent, i.e., varying &theta;, it is then easy to make it closer and closer to &phi;. This is especially true if the dimension of &theta; (i.e., the number of directions one can use to reach &phi;) is increased. This is possible by increasing the size L of the circuit.

For low values of L, one may actually ask whether the found minimum is a global one. I cannot answer this question for sure. I just can notice that, given the symmetry of the circuit with respect to the exchange of any of the qubits, &epsilon; possesses a large number of equivalent minima, and the one encountered within the descent procedure depends on the initial condition for the parameters. Anyway, the actual position of this minimum in &theta;-space is not important.

The main question can be answered by specifying just `Lmax` as a command-line parameter for the script; as explained above, in this situation the arguments `yellow_gate` and `green_gate` default to `x` and `z`. By specifying different values for the remaining parameters one can change the cricuit content, allowing to answer the bonus question. From the reports it is easily argued that any combinations of two different gates leads to a situation similar to the one of the main question, i.e., a convergence of &epsilon; to 0 increasing L: indeed, any couple of different gates makes exactly the same job of `Rx` and `Rz`. By using a couple identical gates for the yellow and green blocks two different situations can raise. If only `Rz` gates are used, the initial `|0000>` state is an eigenstate of the circuit, and therefore no descent is possible; for any L, &epsilon; will not move from its initial value. Instead, if only `x` or `y` gates are used, the convergence of &epsilon; to 0 as a dunction L disppears.