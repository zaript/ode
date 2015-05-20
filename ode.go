// Copyright © 2015 Timur Zaripov. All Rights Reserved.

// Package provides basic support for integration of ODE systems.
//
// Solves the initial value problem for systems of first order ode-s
//        dy/dt = f(y,t0,...)
// where y can be a vector.
//
// Current design:
//
// Problem struct is used to set up calculation.
//
// (p Problem) Solve() returns solution vectors for every given time value.
//
// NewSolverStepXXX functions construct stepping functions from given
// function for right side of system (has YPrime signature) and parameters
// privided by Problem.
//
// Stepping functions perform single integration step for given time t,
// solution vector y and time step dt and return new values of t, y and dt.
//
// Initial version of stepping function design was heavily influenced by
// example from http://rosettacode.org/wiki/Runge-Kutta_method#Go
//
// Description of Runge-Kutta-Merson method can be found here:
// http://www.encyclopediaofmath.org/index.php/Kutta-Merson_method
package ode

import (
	"errors"
	m "math"
)

// Returns solution y of a problem at time points t and an error.
type Solver interface {
	Solve() ([][]float64, error)
}

type Stepper interface {
	Step() error // Perform one step of the method
}

type Problem struct {
	// required parameters
	YP YPrime    // function to compute right side of the system
	Y0 []float64 // initial values
	T  []float64 // time points to find solution at,
	// T[0] = t0, T[len(T)-1:] = tEnd, len(T) >= 2

	// parameters with default values
	Dt0    float64 // initial time step
	ATol   float64 // absolute error tolerance
	RTol   float64 // relative error tolerance
	Safety float64 // safety factor on new step selection
}

type YPrime func(t float64, y []float64) []float64

func (p Problem) Solve() ([][]float64, error) {

	err := p.SanityCheck()
	if err != nil {
		return nil, err
	}

	ys := make([][]float64, len(p.T))
	ys[0] = make([]float64, len(p.Y0))
	copy(ys[0], p.Y0)

	dt := p.Dt0
	t := p.T[0]
	step := NewSolverStepRKM(p.YP, len(p.Y0), p)
	for i := 1; i < len(p.T); i++ {
		y := make([]float64, len(p.Y0))
		// use solution at previous step as new initial value
		copy(y, ys[i-1])
		for t < p.T[i] {
			// we select appropriate dt before step to avoid corner case at
			// t == p.T[i], which turns dt into 0
			dt = m.Min(dt, p.T[i]-t)
			t, y, dt = step(t, y, dt)
		}

		ys[i] = y
	}

	return ys, nil
}

// TODO All sanity checks should emit log messages.
// TODO proper error messages
func (p *Problem) SanityCheck() error {

	if len(p.Y0) == 0 {
		return errors.New("default error message")
	}

	for _, y := range p.Y0 {
		if m.IsNaN(y) || m.IsInf(y, 0) {
			return errors.New("default error message")
		}

	}

	if len(p.T) < 2 {
		return errors.New("default error message")
	}

	for _, t := range p.T {
		if m.IsNaN(t) || m.IsInf(t, 0) {
			return errors.New("NaN or Inf value")
		}
		// if t < p.T[i-1] {
		// 	return errors.New("T is unsorted")
		// }

	}

	if p.Dt0 <= 0 || m.IsNaN(p.Dt0) || m.IsInf(p.Dt0, 0) {
		p.Dt0 = 1e-6 // default time step
	}

	if p.ATol == 0 || m.IsNaN(p.ATol) || m.IsInf(p.ATol, 0) {
		p.ATol = 1e-6 // default atol
	}

	if p.RTol == 0 || m.IsNaN(p.RTol) || m.IsInf(p.RTol, 0) {
		p.RTol = 1e-6 // default rtol
	}

	if p.Safety <= 0 || p.Safety > 1 || m.IsNaN(p.Safety) || m.IsInf(p.Safety, 0) {
		p.Safety = 0.8 // default safety
	}

	if p.YP == nil {
		return errors.New("default error message")
	}

	return nil
}

type solverStep func(t float64, y []float64, dt float64) (float64, []float64, float64)

// Takes a function that calculates y' (y prime) and calculation
// parameters.
// Returns a function that performs a single step of the
// of the classical forth-order Runge-Kutta method.
func NewSolverStepRK4(yp YPrime, nEq int) solverStep {
	// we only want to allocate memory for increments once to avoid excessive gc
	dy1 := make([]float64, nEq)
	dy2 := make([]float64, nEq)
	dy3 := make([]float64, nEq)
	dy4 := make([]float64, nEq)
	// TODO but now len(y) might be different from nEq, check required
	solverStep :=
		func(t float64, y []float64, dt float64) (float64, []float64, float64) {
			copy(dy1, y)
			dy1 = yp(t, dy1)

			copy(dy2, y)
			for iEq := range y {
				dy2[iEq] += dt * dy1[iEq] / 2
			}
			dy2 = yp(t+dt/2, dy2)

			copy(dy3, y)
			for iEq := range y {
				dy3[iEq] += dt * dy2[iEq] / 2
			}
			dy3 = yp(t+dt/2, dy3)

			copy(dy4, y)
			for iEq := range y {
				dy4[iEq] += dt * dy3[iEq]
			}
			dy4 = yp(t+dt, dy4)

			for iEq := range y {
				y[iEq] += dt * (dy1[iEq] + 2*(dy2[iEq]+dy3[iEq]) + dy4[iEq]) / 6
			}
			return t + dt, y, dt
		}
	return solverStep
}

// Takes a function that calculates y' (y prime) and calculation
// parameters.
// Returns a function that performs a single step of the
// Runge-Kutta-Merson method with adaptive step.
func NewSolverStepRKM(yp YPrime, nEq int, p Problem) solverStep {
	// we only want to allocate memory for increments once to avoid excessive gc
	dy1 := make([]float64, nEq)
	dy2 := make([]float64, nEq)
	dy3 := make([]float64, nEq)
	dy4 := make([]float64, nEq)
	dy5 := make([]float64, nEq)
	// TODO but now len(y) might be different from nEq, check required
	solverStep :=
		func(t float64, y []float64, dt float64) (float64, []float64, float64) {
			copy(dy1, y)
			dy1 = yp(t, dy1)

			copy(dy2, y)
			for iEq := range y {
				dy2[iEq] += dt * dy1[iEq] / 3
			}
			dy2 = yp(t+dt/3, dy2)

			copy(dy3, y)
			for iEq := range y {
				dy3[iEq] += (dy1[iEq] + dy2[iEq]) * dt / 6
			}
			dy3 = yp(t+dt/3, dy3)

			copy(dy4, y)
			for iEq := range y {
				dy4[iEq] += (dy1[iEq] + 3*dy3[iEq]) * dt / 8
			}
			dy4 = yp(t+dt/2, dy4)

			copy(dy5, y)
			for iEq := range y {
				dy5[iEq] += (dy1[iEq] - 3*dy3[iEq] + 4*dy4[iEq]) * dt / 2
			}
			dy5 = yp(t+dt, dy5)

			// Pick highest estimation of truncation error among all equations.
			err := 0.
			for iEq := range y {
				tmp := m.Abs((2*dy1[iEq] - 9*dy3[iEq] + 8*dy4[iEq] - dy5[iEq]) * dt / 30)
				err = m.Max(err, tmp)
			}

			// If error is in tolerance interval proceed with the step.
			if err < p.ATol {
				for iEq := range y {
					y[iEq] += (dy1[iEq] + 4*dy4[iEq] + dy5[iEq]) * dt / 6
				}
				t += dt
			}

			// TODO this is really bad, need to rethink this check
			if err != 0 {
				dt = p.Safety * dt * m.Pow(p.ATol/err, 0.2)
			}

			return t, y, dt
		}
	return solverStep
}
