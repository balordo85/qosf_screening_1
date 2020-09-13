import math, random, datetime, sys
import numpy, qiskit, scipy, matplotlib

#############################
#    CONSTANTS              #
#############################

# Generation of a random state of 4 qubits
phi = qiskit.quantum_info.random_statevector((2, 2, 2, 2))

#############################
#    DEFINITIONS            #
#############################

def argv_validation():
    '''
    argv detection
    First argument (mandatory): Lmax. Allowed values: integer larger than 1
    Second and third arguments (optional): gates building yellow and green blocks respectively. Allowed values: 'x', 'y', 'z'. Default: 'x' and 'z'
    '''
    
    Lmax = None
    yellow_gate = None
    green_gate = None
    
    if len(sys.argv) == 2:
        if int(sys.argv[1]) <= 1:
            raise Exception('Invalid value for Lmax')
        Lmax = int(sys.argv[1])

        yellow_gate = 'x'
        green_gate = 'z'
    elif len(sys.argv) == 4:
        if int(sys.argv[1]) <= 1:
            raise Exception('Invalid value for Lmax')
        Lmax = int(sys.argv[1])

        if not(sys.argv[2] in ['x', 'y', 'z']):
            raise Exception('Invalid value for yellow_gate')
        yellow_gate = sys.argv[2] 

        if not(sys.argv[3] in ['x', 'y', 'z']):
            raise Exception('Invalid value for green_gate')
        green_gate = sys.argv[3]
    else:
        raise Exception('Invalid argument number')
        
    return (Lmax, yellow_gate, green_gate)


def circuit(theta):
    '''
    Circuit-constructing function
    '''

    # Getting L as (# of columns of theta) / 8
    L = int(theta.shape[0] / 8)
    
    # Building the circuit
    toBeReturned = yellow([theta[0], theta[1], theta[2], theta[3]], yellow_gate)
    for l in range(1, L + 1):
        if l > 1:
            toBeReturned = toBeReturned.compose(yellow([theta[8 * (l - 1)], theta[8 * (l - 1) + 1], theta[8 * (l - 1) + 2], theta[8 * (l - 1) + 3]], yellow_gate))
        toBeReturned = toBeReturned.compose(green([theta[8 * (l - 1) + 4], theta[8 * (l - 1) + 5], theta[8 * (l - 1) + 6], theta[8 * (l - 1) + 7]], green_gate))
        
    return toBeReturned

def yellow(theta, gate):
    '''
    Circuit building blocks (odd)
    '''

    toBeReturned = qiskit.QuantumCircuit(4)
    
    op = {
        'x': toBeReturned.rx,
        'y': toBeReturned.ry,
        'z': toBeReturned.rz
    }
    op[gate](theta[0], 0)
    op[gate](theta[1], 1)
    op[gate](theta[2], 2)
    op[gate](theta[3], 3)
    
    return toBeReturned

def green(theta, gate):
    '''
    Circuit building blocks (even)
    '''

    toBeReturned =  yellow(theta, gate)
    
    toBeReturned.cz(0, 1)
    toBeReturned.cz(0, 2)
    toBeReturned.cz(0, 3)
    toBeReturned.cz(1, 2)
    toBeReturned.cz(1, 3)
    toBeReturned.cz(2, 3)
    
    return toBeReturned

def norm(state):
    '''
    Norm of a Statevector(.data)
    '''

    toBeReturned = 0
    for j in range(2**4):
        toBeReturned = toBeReturned + state[j].real**2 + state[j].imag**2
    return toBeReturned

def distance(theta, phi):    
    '''
    Distance between the output of the circuit at some theta and a generic Statevector
    '''

    # Simulator choice
    simulator = qiskit.Aer.get_backend('statevector_simulator')
    
    # Circuit simulation
    result = qiskit.execute(circuit(theta), simulator).result()
    
    # Result Statevector
    result_statevector = result.get_statevector()
    
    # Returning the result
    return norm(result_statevector.data - phi.data)

def dist(theta):
    '''
    Distance as a function of theta
    '''
    return distance(theta, phi)
    
def date2Str():
    '''
    Datetime string
    '''
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    
def generate_report():
    '''
    Report generation
    '''

    # Common file name
    filename = 'report/Lmax{}_yg{}_gg{}_{}'.format(str(Lmax), yellow_gate, green_gate, date2Str())

    # Plot
    output_x = range(1, Lmax + 1)
    matplotlib.pyplot.axis([1, Lmax, 0, round(max(output_y), 1) + 0.1])
    matplotlib.pyplot.xticks(output_x)
    matplotlib.pyplot.xlabel('L')
    matplotlib.pyplot.ylabel('\u03B5')
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.text(1.5, round(max(output_y), 1), 'Details in the accompanying text file\n yellow_gate = {}, green_gate = {}'.format(yellow_gate, green_gate))
    axes = matplotlib.pyplot.plot(output_x, output_y, 'ro')
    axes[0].set_clip_on(False)
    matplotlib.pyplot.savefig(filename + '.png')
    
    # Text file
    file = open(filename + '.txt', 'x')
    file.write('phi = {}\n\n'.format(str(phi)))
    for L in output_x:
        file.write('L = {}:\n'.format(L))
        file.write('theta0 = {}\n'.format(initial_theta[L - 1]))
        file.write('theta_result = {}\n'.format(final_theta[L - 1]))
        file.write('epsilon = {}\n'.format(output_y[L - 1]))
        file.write('Function evaluations: {}\n'.format(func_calls[L - 1]))
        file.write('Gradient evaluations: {}\n'.format(grad_calls[L - 1]))
        file.write('Warning: {} (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_cg.html for details)\n\n'.format(warnflag[L - 1]))

#############################
#    MAIN                   #
#############################

if __name__ == '__main__':
    # argv
    Lmax, yellow_gate, green_gate = argv_validation()

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

        # circuit(theta0).draw(output = 'mpl', filename = 'report/pippo_{}.png'.format(L))
        
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