package ode

import (
	"fmt"
	"math"
	"testing"
)

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
func yPrimeExpODE(t float64, y []float64) []float64 {
	y[0] = a0 * y[0]
	y[1] = a1 * y[1]
	y[2] = a2 * y[2]
	return y
}

// with exact solution
// u_i = exp(t*A)*u_i(0) for i=0..2
func yExactExpODE(t float64, nEq int) []float64 {
	y := make([]float64, nEq)
	y[0] = math.Exp(t*a0) * y0_0
	y[1] = math.Exp(t*a1) * y1_0
	y[2] = math.Exp(t*a2) * y2_0
	return y
}

// Numerical and exact solutions are compared at integer values of time
// (1, 2, 3, etc). Test fails if difference is higher than given tolerance.
func Test_Solve_expODE(test *testing.T) {
	p := Problem{
		YP:     yPrimeExpODE,
		Y0:     []float64{y0_0, y1_0, y2_0},
		T:      []float64{0, 1, 2, 3, 4, 5},
		Dt0:    1.1e-3,
		ATol:   1.1e-5,
		RTol:   1.1e-5,
		Safety: 0.8,
	}

	y, err := p.Solve()
	if err != nil {
		test.Error("Solve(): expODE failed.")
	}

	for i, t := range p.T {
		yRef := yExactExpODE(t, len(p.Y0))
		if !allWithinTol(y[i], yRef, p.RTol, p.ATol) {
			fmt.Println("yNum =", y[i])
			fmt.Println("yRef =", yRef)
			test.Error("Solve(): expODE failed.")
		}
	}
}

func Test_NewSolverStepRKM_expODE(test *testing.T) {
	p := Problem{
		YP:     yPrimeExpODE,
		Y0:     []float64{y0_0, y1_0, y2_0},
		T:      []float64{0, 1},
		Dt0:    1.1e-3,
		ATol:   1.1e-5,
		RTol:   1.1e-5,
		Safety: 0.8,
	}

	y := p.Y0
	t := p.T[0]
	tEnd := p.T[len(p.T)-1]
	tCheck := 1.
	dt := p.Dt0

	solverStep := NewSolverStepRKM(yPrimeExpODE, p)
	for t < tEnd {
		dt = math.Min(dt, tCheck-t)
		t, y, dt = solverStep(t, y, dt)

		if t >= tCheck {
			yRef := yExactExpODE(t, len(p.Y0))
			if !allWithinTol(y, yRef, p.RTol, p.ATol) {
				fmt.Println("yNum =", y)
				fmt.Println("yRef =", yRef)
				test.Error("RKM: expODE failed.")
			}
			tCheck++
		}
	}
}

func Test_NewSolverStepRK4_expODE(test *testing.T) {
	p := Problem{
		YP:     yPrimeExpODE,
		Y0:     []float64{y0_0, y1_0, y2_0},
		T:      []float64{0, 1},
		Dt0:    1.1e-3,
		ATol:   1.1e-4,
		RTol:   1.1e-4,
		Safety: 0.8,
	}

	y := p.Y0
	t := p.T[0]
	tEnd := p.T[len(p.T)-1]
	tCheck := 1.
	dt := p.Dt0

	solverStep := NewSolverStepRK4(yPrimeExpODE, p)
	for t < tEnd {
		dt = math.Min(dt, tCheck-t)
		t, y, dt = solverStep(t, y, dt)

		if t >= tCheck {
			yRef := yExactExpODE(t, len(p.Y0))
			if !allWithinTol(y, yRef, p.RTol, p.ATol) {
				fmt.Println("yNum =", y)
				fmt.Println("yRef =", yRef)
				test.Error("RKM: expODE failed.")
			}
			tCheck++
			dt = p.Dt0
		}
	}
}

// Checks if a and b are equal within a tolerance.
// TODO NaN and Inf checks
func withinTol(a, b, rtol, atol float64) bool {
	return math.Abs(a-b) <= atol+rtol*math.Abs(b)
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
