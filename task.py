import math, random, datetime, sys
import numpy, qiskit, scipy, matplotlib

#############################
#    DEFINITIONS            #
#############################

# argv detection
# First argument (mandatory): Lmax. Allowed values: integer larger than 1
# Second and third arguments (optional): gates building yellow and green blocks respectively. Allowed values: 'x', 'y', 'z'. Default: 'x' and 'z'
def argv_validation():
    Lmax = None
    yellow_gate = None
    green_gate = None
    
    if len(sys.argv) == 1:
        raise Exception('Specify at least Lmax')
    elif len(sys.argv) == 2:
        if int(sys.argv[1]) <= 1:
            raise Exception('Invalid value for Lmax')
        else:
            Lmax = int(sys.argv[1])
            
        yellow_gate = 'x'
        green_gate = 'z'
    elif len(sys.argv) == 4:
        if int(sys.argv[1]) <= 1:
            raise Exception('Invalid value for Lmax')
        else:
            Lmax = int(sys.argv[1])
            
        if not(sys.argv[2] == 'x' or sys.argv[2] == 'y' or sys.argv[2] == 'z'):
            raise Exception('Invalide value for yellow_gate')
        else:
            yellow_gate = sys.argv[2]
            
        if not(sys.argv[3] == 'x' or sys.argv[3] == 'y' or sys.argv[3] == 'z'):
            raise Exception('Invalide value for green_gate')
        else:
            green_gate = sys.argv[3]
    else:
        raise Exception('Invalid argument number')
        
    return (Lmax, yellow_gate, green_gate)

# Circuit-constructing function
def circuit(theta):
    # Getting L as (# of columns of theta) / 8
    L = int(theta.shape[0] / 8)
    
    # Building the circuit
    toBeReturned = yellow([theta[0], theta[1], theta[2], theta[3]], yellow_gate)
    for l in range(1, L + 1):
        if l > 1:
            toBeReturned = toBeReturned.compose(yellow([theta[8 * (l - 1)], theta[8 * (l - 1) + 1], theta[8 * (l - 1) + 2], theta[8 * (l - 1) + 3]], yellow_gate))
        toBeReturned = toBeReturned.compose(green([theta[8 * (l - 1) + 4], theta[8 * (l - 1) + 5], theta[8 * (l - 1) + 6], theta[8 * (l - 1) + 7]], green_gate))
        
    return toBeReturned

# Circuit building blocks
def green(theta, gate):
    toBeReturned = qiskit.QuantumCircuit(4)
    
    if 'x' == gate:
        toBeReturned.rx(theta[0], 0)
        toBeReturned.rx(theta[1], 1)
        toBeReturned.rx(theta[2], 2)
        toBeReturned.rx(theta[3], 3)
    elif 'y' == gate:
        toBeReturned.ry(theta[0], 0)
        toBeReturned.ry(theta[1], 1)
        toBeReturned.ry(theta[2], 2)
        toBeReturned.ry(theta[3], 3)
    else:
        toBeReturned.rz(theta[0], 0)
        toBeReturned.rz(theta[1], 1)
        toBeReturned.rz(theta[2], 2)
        toBeReturned.rz(theta[3], 3)
    
    toBeReturned.cz(0, 1)
    toBeReturned.cz(0, 2)
    toBeReturned.cz(0, 3)
    toBeReturned.cz(1, 2)
    toBeReturned.cz(1, 3)
    toBeReturned.cz(2, 3)
    
    return toBeReturned
def yellow(theta, gate):
    toBeReturned = qiskit.QuantumCircuit(4)
    
    if 'y' == gate:
        toBeReturned.ry(theta[0], 0)
        toBeReturned.ry(theta[1], 1)
        toBeReturned.ry(theta[2], 2)
        toBeReturned.ry(theta[3], 3)
    elif 'z' == gate:
        toBeReturned.rz(theta[0], 0)
        toBeReturned.rz(theta[1], 1)
        toBeReturned.rz(theta[2], 2)
        toBeReturned.rz(theta[3], 3)
    else:
        toBeReturned.rx(theta[0], 0)
        toBeReturned.rx(theta[1], 1)
        toBeReturned.rx(theta[2], 2)
        toBeReturned.rx(theta[3], 3)
    
    return toBeReturned

# Norm of a Statevector(.data)
def norm(state):
    toBeReturned = 0
    for j in range(2**4):
        toBeReturned = toBeReturned + state[j].real**2 + state[j].imag**2
    return toBeReturned

# Distance between the output of the circuit at some theta and a generic Statevector
def distance(theta, phi):    
    # Simulator choice
    simulator = qiskit.Aer.get_backend('statevector_simulator')
    
    # Circuit simulation
    result = qiskit.execute(circuit(theta), simulator).result()
    
    # Result Statevector
    result_statevector = result.get_statevector()
    
    # Returning the result
    return norm(result_statevector.data - phi.data)

# Distance as a function of theta
def dist(theta):
    return distance(theta, phi)
    
# Datetime string
def date2Str():
    date = datetime.datetime.now()
    year = date.strftime('%Y')
    month = date.strftime('%m')
    day = date.strftime('%d')
    hour = date.strftime('%H')
    minute = date.strftime('%M')
    second = date.strftime('%S')
    return year + month + day + hour + minute + second
    
# Report generation
def generate_report():
    # Plot
    output_x = range(1, Lmax + 1)
    matplotlib.pyplot.axis([1, Lmax, 0, round(max(output_y), 1) + 0.1])
    matplotlib.pyplot.xticks(output_x)
    matplotlib.pyplot.xlabel('L')
    matplotlib.pyplot.ylabel('\u03B5')
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.text(1.5, round(max(output_y), 1), 'Details in the accompanying text file\n yellow_gate = ' + yellow_gate + ', green_gate = ' + green_gate)
    axes = matplotlib.pyplot.plot(output_x, output_y, 'ro')
    axes[0].set_clip_on(False)
    matplotlib.pyplot.savefig('report/Lmax' + str(Lmax) + '_yg' + yellow_gate + '_gg' + green_gate + '_' + date2Str() + '.png')
    
    # Text file
    file = open('report/Lmax' + str(Lmax) + '_yg' + yellow_gate + '_gg' + green_gate + '_' + date2Str() + '.txt', 'x')
    file.write('phi = ' + str(phi) + '\n\n')
    for L in output_x:
        file.write('L = ' + str(L) + ':\n')
        file.write('theta0 = ' + str(initial_theta[L - 1]) + '\n')
        file.write('theta_result = ' + str(final_theta[L - 1]) + '\n')
        file.write('epsilon = ' + str(output_y[L - 1]) + '\n')
        file.write('Function evaluations: ' + str(func_calls[L - 1]) + '\n')
        file.write('Gradient evaluations: ' + str(grad_calls[L - 1]) + '\n')
        file.write('Warning: ' + str(warnflag[L - 1]) + ' (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_cg.html for details)\n\n')

#############################
#    MAIN                   #
#############################

# argv
Lmax, yellow_gate, green_gate = argv_validation()

# Generation of a random state of 4 qubits
phi = qiskit.quantum_info.random_statevector((2, 2, 2, 2))

# Initial and final values for theta (useful for the report)
initial_theta = [None] * Lmax
final_theta = [None] * Lmax

# Output of the computation
output_y = [None] * Lmax
func_calls = [None] * Lmax
grad_calls = [None] * Lmax
warnflag = [None] * Lmax

# Simulation for L = 1, ..., Lmax
for L in range(1, Lmax + 1):
    print('L = ' + str(L))
    
    theta0 = numpy.zeros(4 * 2 * L)
    for j in range (4 * 2 * L):
        theta0[j] = random.random() * 2 * math.pi
    
    # Conjugate gradient descent
    result = scipy.optimize.fmin_cg(f = dist, x0 = theta0, full_output = True)
    
    # Output
    initial_theta[L - 1] = theta0
    final_theta[L - 1] = result[0]
    output_y[L - 1] = result[1]
    func_calls[L - 1] = result[2]
    grad_calls[L - 1] = result[3]
    warnflag[L - 1] = result[4]
    
# Report generation
generate_report()