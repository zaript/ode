package ode

import (
	"fmt"
	m "math"
	"testing"
)

// Structure to contain calculation parameters.
type prm struct {
	nEq    uint64  // number of equations
	dt     float64 // initial time step
	atol   float64 // absolute error tolerance
	rtol   float64 // relative error tolerance
	safety float64 // safety factor on new step selection
	tStart float64 // starting time (usually 0)
	tEnd   float64 // finish time (maximal value of t)
	ndt    uint64  // maximal number of time steps
}

// prm should implement SolverPrm interface.
func (p prm) NEq() uint64     { return p.nEq }
func (p prm) Tol() float64    { return p.rtol }
func (p prm) TEnd() float64   { return p.tEnd }
func (p prm) Safety() float64 { return p.safety }

// Coefficients and initial values used in simple ODE with 3 independent
// equations which is used to test solvers.
const (
	a0   = 1.0
	a1   = 2.0
	a2   = 2.0
	y0_0 = 1.0
	y1_0 = 1.0
	y2_0 = 2.0
)

// System to solve looks like:
// du_i/dt = A*u_i for i=0..2
func yPrimeExpODE(t float64, y []float64, prm SolverPrm) []float64 {
	y[0] = a0 * y[0]
	y[1] = a1 * y[1]
	y[2] = a2 * y[2]
	return y
}

// with exact solution
// u_i = exp(t*A)*u_i(0) for i=0..2
func yExactExpODE(t float64, prm prm) []float64 {
	y := make([]float64, prm.nEq)
	y[0] = m.Exp(t*a0) * y0_0
	y[1] = m.Exp(t*a1) * y1_0
	y[2] = m.Exp(t*a2) * y2_0
	return y
}

// Difference between numerical and exact solutions is compared to expected error
// at integer values of time (1, 2, 3, etc). Test fails if an actual error is
// higher than given tolerance*10.
func Test_NewSolverStepRKM_expODE(test *testing.T) {
	prm := prm{
		atol:   1e-4,
		rtol:   1e-4,
		safety: 0.8,
		dt:     1e-4,
		tStart: 0.,
		tEnd:   1.,
		ndt:    300,
		nEq:    3,
	}

	y := make([]float64, prm.nEq)
	// set initial conditions
	y[0] = y0_0
	y[1] = y1_0
	y[2] = y2_0

	t := prm.tStart
	dt := prm.dt

	// prm.tEnd in this example is always set to next integer value
	// so we need additional variable for an actual end time.
	tEnd := 10.0
	dtLast := dt

	solverStep := NewSolverStepRKM(yPrimeExpODE, prm)
	for iT := uint64(0); iT < prm.ndt; iT++ {
		t, y, dt = solverStep(t, y, dt)

		// dt == 0 means that we reached prm.tEnd.
		if dt == 0 {
			y_exact := yExactExpODE(t, prm)
			for i := range y {
				e := m.Abs(y[i] - y_exact[i])
				if e >= prm.atol*10 {
					test.Error("t =", t, "iT =", iT, "e =", e, "tol =", prm.atol,
						"i =", i, "y =", y[i], "y_exact =", y_exact[i])
				}
			}

			prm.tEnd += 1.
			dt = dtLast
		} else {
			dtLast = dt
		}

		if prm.tEnd > tEnd {
			break
		}
	}
}

// Test of fourth order Runge-Kutta solver.
//
// Difference between numerical and exact solutions is compared to expected error
// at integer values of time (1, 2, 3, etc). Test fails if an actual error is
// higher than given tolerance.
func Test_NewSolverStepRK4_expODE(test *testing.T) {
	prm := prm{
		atol:   1e-4,
		rtol:   1e-4,
		safety: 0.0,
		dt:     1e-3,
		tStart: 0.,
		tEnd:   1.,
		ndt:    3000,
		nEq:    3,
	}

	y := make([]float64, prm.nEq)
	// set initial conditions
	y[0] = y0_0
	y[1] = y1_0
	y[2] = y2_0

	t := prm.tStart
	dt := prm.dt

	// prm.tEnd in this example is used to calculate solution at integer steps
	// so we need additional variable for an actual end time.
	tEnd := 10.0

	solverStep := NewSolverStepRK4(yPrimeExpODE, prm)
	for iT := uint64(0); iT < prm.ndt; iT++ {
		t, y = solverStep(t, y, dt)

		if t >= prm.tEnd {
			y_exact := yExactExpODE(t, prm)
			for i := range y {
				e := m.Abs(y[i] - y_exact[i])
				if e >= prm.atol*10 {
					test.Error("t =", t, "iT =", iT, "e =", e, "tol =", prm.atol,
						"i =", i, "y =", y[i], "y_exact =", y_exact[i])
				}
			}

			prm.tEnd += 1.
		}

		if prm.tEnd > tEnd {
			break
		}
	}
}

// we want all benchmarks below to have same calculation parameters
// for comparison
func (p prm) initBench() prm {
	p = prm{
		atol:   1e-6,
		rtol:   1e-6,
		safety: 0.8,
		dt:     1e-6,
		tStart: 0.,
		tEnd:   10.,
		ndt:    50000,
		nEq:    3,
	}
	return p
}

func Benchmark_NewSolverStepRKM_expODE(b *testing.B) {
	for iTest := 0; iTest < b.N; iTest++ {
		var prm prm
		prm = prm.initBench()

		y := make([]float64, prm.nEq)
		// set initial conditions
		y[0] = y0_0
		y[1] = y1_0
		y[2] = y2_0

		t := prm.tStart
		dt := prm.dt

		solverStep := NewSolverStepRKM(yPrimeExpODE, prm)
		for iT := uint64(0); iT < prm.ndt; iT++ {
			t, y, dt = solverStep(t, y, dt)

			if t >= prm.tEnd {
				break
			}
		}
	}
}

func Benchmark_NewSolverStepRK4_expODE(b *testing.B) {
	for iTest := 0; iTest < b.N; iTest++ {
		var prm prm
		prm = prm.initBench()

		y := make([]float64, prm.nEq)
		// set initial conditions
		y[0] = y0_0
		y[1] = y1_0
		y[2] = y2_0

		t := prm.tStart
		dt := prm.dt

		solverStep := NewSolverStepRK4(yPrimeExpODE, prm)
		for iT := uint64(0); iT < prm.ndt; iT++ {
			t, y = solverStep(t, y, dt)

			if t >= prm.tEnd {
				break
			}
		}
	}
}

// Set of problems used for testing below composed by:
// Mazzia, Francesca and Magherini, Cecilia,
// Test Set for Initial Value Problem Solvers, release 2.4, February 2008,
// Department of Mathematics, University of Bari and INdAM, Research Unit of Bari,
// Available at http://www.dm.uniba.it/~testset.

// Checks if a and b are equal within a tolerance.
// TODO NaN and Inf checks
func withinTol(a, b, rtol, atol float64) bool {
	return m.Abs(a-b) <= atol+rtol*m.Abs(b)
}

// Checks if all elements of slices a and b are equal within a tolerance.
func allWithinTol(a, b []float64, rtol, atol float64) bool {
	if a == nil || b == nil || len(a) != len(b) {
		return false
	}
	tmp := true
	for i := 0; (i < len(a)) && tmp; i++ {
		tmp = tmp && withinTol(a[i], b[i], rtol, atol)
	}
	return tmp
}

// HIRES problem (ODE)
func Test_HIRES(test *testing.T) {
	prm := prm{
		atol:   1.1e-6,
		rtol:   1.1e-6,
		safety: 0.8,
		dt:     1.1e-3,
		tStart: 0.,
		tEnd:   321.8122,
		ndt:    50000,
		nEq:    8,
	}
	y0 := []float64{1, 0, 0, 0, 0, 0, 0, 0.0057}
	yRef := []float64{
		0.7371312573325668e-3,
		0.1442485726316185e-3,
		0.5888729740967575e-4,
		0.1175651343283149e-2,
		0.2386356198831331e-2,
		0.6238968252742796e-2,
		0.2849998395185769e-2,
		0.2850001604814231e-2,
	}
	hiresYPrime :=
		func(t float64, y []float64, prm SolverPrm) []float64 {
			f := make([]float64, len(y))
			f[0] = -1.71*y[0] + 0.43*y[1] + 8.32*y[2] + 0.0007
			f[1] = 1.71*y[0] - 8.75*y[1]
			f[2] = -10.03*y[2] + 0.43*y[3] + 0.035*y[4]
			f[3] = 8.32*y[1] + 1.71*y[2] - 1.12*y[3]
			f[4] = -1.745*y[4] + 0.43*(y[5]+y[6])
			f[5] = -280*y[5]*y[7] + 0.69*y[3] + 1.71*y[4] - 0.43*y[5] + 0.69*y[6]
			f[6] = 280*y[5]*y[7] - 1.81*y[6]
			f[7] = -f[6]
			return f
		}

	adaptiveStep := NewSolverStepRKM(hiresYPrime, prm)

	y := make([]float64, prm.nEq)
	copy(y, y0)
	t := prm.tStart
	dt := prm.dt
	for t < prm.tEnd {
		t, y, dt = adaptiveStep(t, y, dt)
	}
	if !allWithinTol(y, yRef, prm.rtol, prm.atol) {
		// TODO turn this into nice log output
		fmt.Println("y =", y)
		fmt.Println("yRef =", yRef)
		test.Error("RKM: HIRES failed.")
	}

	// TODO Should probably be in separate function, but I want to avoid code
	// duplication until things stabilize a bit
	fixedStep := NewSolverStepRK4(hiresYPrime, prm)

	copy(y, y0)
	t = prm.tStart
	dt = prm.dt
	for t < prm.tEnd {
		t, y = fixedStep(t, y, dt)
	}
	if !allWithinTol(y, yRef, prm.rtol, prm.atol) {
		// TODO turn this into nice log output
		fmt.Println("y =", y)
		fmt.Println("yRef =", yRef)
		test.Error("RK4: HIRES failed.")
	}
}
