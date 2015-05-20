package ode

import (
	"fmt"
	"math"
	"testing"
)

// we want all benchmarks below to have same calculation parameters
// for comparison
func initBench() Problem {
	p := Problem{
		YP:     yPrimeExpODE,
		Y0:     []float64{y0_0, y1_0, y2_0},
		T:      []float64{0, 1, 2, 3, 4, 5},
		Dt0:    1.1e-3,
		ATol:   1.1e-5,
		RTol:   1.1e-5,
		Safety: 0.8,
	}
	return p
}

func Benchmark_NewSolverStepRKM_expODE(b *testing.B) {
	for iTest := 0; iTest < b.N; iTest++ {
		p := initBench()

		y := p.Y0
		t := p.T[0]
		tEnd := p.T[len(p.T)-1]
		dt := p.Dt0

		solverStep := NewSolverStepRKM(yPrimeExpODE, p)
		for t < tEnd {
			dt = math.Min(dt, tEnd-t)
			t, y, dt = solverStep(t, y, dt)
		}
	}
}

func Benchmark_NewSolverStepRKM_flex_expODE(b *testing.B) {
	for iTest := 0; iTest < b.N; iTest++ {
		p := initBench()

		y := p.Y0
		t := p.T[0]
		tEnd := p.T[len(p.T)-1]
		dt := p.Dt0

		solverStep := newSolverStepRKM_flex(yPrimeExpODE, p)
		for t < tEnd {
			dt = math.Min(dt, tEnd-t)
			t, y, dt = solverStep(t, y, dt)
		}
	}
}

func Benchmark_NewSolverStepRK4_expODE(b *testing.B) {
	for iTest := 0; iTest < b.N; iTest++ {
		p := initBench()

		y := p.Y0
		t := p.T[0]
		tEnd := p.T[len(p.T)-1]
		dt := p.Dt0

		solverStep := NewSolverStepRK4(yPrimeExpODE, p)
		for t < tEnd {
			dt = math.Min(dt, tEnd-t)
			t, y, dt = solverStep(t, y, dt)
		}
	}
}

func Benchmark_NewSolverStepRK4_flex_expODE(b *testing.B) {
	for iTest := 0; iTest < b.N; iTest++ {
		p := initBench()

		y := p.Y0
		t := p.T[0]
		tEnd := p.T[len(p.T)-1]
		dt := p.Dt0

		solverStep := newSolverStepRK4_flex(yPrimeExpODE, p)
		for t < tEnd {
			dt = math.Min(dt, tEnd-t)
			t, y, dt = solverStep(t, y, dt)
		}
	}
}

// Numerical and exact solutions are compared at integer values of time
// (1, 2, 3, etc). Test fails if difference is higher than given tolerance.
func Test_NewSolverStepRKM_flex_expODE(test *testing.T) {
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

	solverStep := newSolverStepRKM_flex(yPrimeExpODE, p)
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

func Test_NewSolverStepRK4_flex_expODE(test *testing.T) {
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

	solverStep := newSolverStepRK4_flex(yPrimeExpODE, p)
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

// Methods below are implemented to benchmark naive RK implementations against
// flexible ones (using Butcher tableau).

func newSolverStepRK4_flex(yp YPrime, p Problem) solverStep {
	// Runge-Kutta matrix
	a := [4][3]float64{
		{0, 0, 0},
		{0.5, 0, 0},
		{0, 0.5, 0},
		{0, 0, 1.},
	}
	// weigths
	b := [4]float64{(1. / 6.), (1. / 3.), (1. / 3.), (1. / 6.)}
	// nodes
	c := [4]float64{0, 0.5, 0.5, 1.}

	// we only want to allocate memory for increments once to avoid excessive gc
	dy := make([][]float64, len(c))
	for i := range c {
		dy[i] = make([]float64, len(p.Y0))
	}

	solverStep :=
		func(t float64, y []float64, dt float64) (float64, []float64, float64) {
			copy(dy[0], y)
			dy[0] = yp(t, dy[0])
			for i := 1; i < len(c); i++ {
				copy(dy[i], y)
				for iEq := range y {
					for j := 0; j < i; j++ {
						dy[i][iEq] += a[i][j] * dy[j][iEq] * dt
					}
				}
				dy[i] = yp(t+dt*c[i], dy[i])
			}

			for iEq := range y {
				for i := range b {
					y[iEq] += dt * dy[i][iEq] * b[i]
				}
			}

			return t + dt, y, dt
		}
	return solverStep
}

func newSolverStepRKM_flex(yp YPrime, p Problem) solverStep {
	// Runge-Kutta matrix
	a := [5][4]float64{
		{0, 0, 0, 0},
		{1. / 3., 0, 0, 0},
		{1. / 6., 1. / 6., 0, 0},
		{1. / 8., 0, 3. / 8., 0},
		{0.5, 0, -3. / 2., 2.},
	}
	// weigths
	b := [5]float64{1. / 6., 0., 0., 2. / 3., 1. / 6.}
	// nodes
	c := [5]float64{0, 1. / 3., 1. / 3., 1. / 2., 1.}
	// error estimation coefficients
	e := [5]float64{1. / 15., 0., -0.3, 4. / 15., -1. / 30.}

	// we only want to allocate memory for increments once to avoid excessive gc
	dy := make([][]float64, len(c))
	for i := range c {
		dy[i] = make([]float64, len(p.Y0))
	}

	solverStep :=
		func(t float64, y []float64, dt float64) (float64, []float64, float64) {
			copy(dy[0], y)
			dy[0] = yp(t, dy[0])
			for i := 1; i < len(c); i++ {
				copy(dy[i], y)
				for iEq := range y {
					for j := 0; j < i; j++ {
						dy[i][iEq] += a[i][j] * dy[j][iEq] * dt
					}
				}
				dy[i] = yp(t+dt*c[i], dy[i])
			}

			// Pick highest estimation of trunkation error among all equations.
			err := 0.
			for iEq := range y {
				tmp := 0.
				for i := range e {
					tmp += dy[i][iEq] * e[i]
				}
				tmp = math.Abs(tmp * dt)
				err = math.Max(err, tmp)
			}

			// If error is in tolerance interval proceed with the step.
			if err < p.ATol {
				for iEq := range y {
					for i := range b {
						y[iEq] += dt * dy[i][iEq] * b[i]
					}
				}
				t += dt
			}

			dt = (p.Safety * dt * math.Pow(p.RTol/err, 0.2))

			return t, y, dt
		}
	return solverStep
}
