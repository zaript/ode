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
// privided by Problemath.
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
	"math"
)

// Returns solution y of a problem at time points t and an error.
type Solver interface {
	Solve(p Problem, prm Parameters) ([][]float64, error)
}

type Stepper interface {
	Init(p *Problem, prm *Parameters)
	// Step return values are new t, y and error estimate.
	Step(t float64, y []float64, dt float64) (float64, []float64, float64)
}

type Adapter interface {
	Init(p *Problem, prm *Parameters)
	// Adapt returns new value of dt.
	Adapt(dt, e float64) float64
}

type YPrime func(t float64, y []float64) []float64

type Problem struct {
	YP   YPrime    // function to compute right side of the system
	Y0   []float64 // initial values
	T0   float64   // starting time
	TEnd float64   // finish time
}

type Parameters struct {
	Dt0    float64 // initial time step
	ATol   float64 // absolute error tolerance
	RTol   float64 // relative error tolerance
	Safety float64 // safety factor on new step selection
}

func Solve(p Problem, prm Parameters) ([]float64, error) {
	// TODO shouldn't this panic?
	err := p.SanityCheck()
	if err != nil {
		return nil, err
	}
	err = prm.SanityCheck()
	if err != nil {
		return nil, err
	}

	s := new(RKM)
	s.Init(&p, &prm)

	a := new(BasicAdaptor)
	a.Init(&p, &prm)

	y := make([]float64, len(p.Y0))
	copy(y, p.Y0)

	dt := prm.Dt0
	t := p.T0
	e := 0.0
	for t < p.TEnd {
		// we select appropriate dt before step to avoid corner case at
		// t == p.T[i], which turns dt into 0
		dt = math.Min(dt, p.TEnd-t)

		t, y, e = s.Step(t, y, dt)
		if e > 0 {
			dt = a.Adapt(dt, e)
		}

	}

	return y, nil
}

// TODO All sanity checks should emit log messages.
// TODO proper error messages
func (p *Problem) SanityCheck() error {
	if len(p.Y0) == 0 {
		return errors.New("default error message")
	}

	for _, y := range p.Y0 {
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return errors.New("default error message")
		}
	}

	if p.YP == nil {
		return errors.New("default error message")
	}

	if math.IsNaN(p.T0) || math.IsInf(p.T0, 0) {
		return errors.New("NaN or Inf value")
	}

	if math.IsNaN(p.TEnd) || math.IsInf(p.TEnd, 0) {
		return errors.New("NaN or Inf value")
	}

	if p.T0 > p.TEnd {
		return errors.New("T is unsorted")
	}

	return nil
}

func (p *Parameters) SanityCheck() error {
	if p.Dt0 <= 0 || math.IsNaN(p.Dt0) || math.IsInf(p.Dt0, 0) {
		p.Dt0 = 1e-6 // default time step
	}

	if p.ATol == 0 || math.IsNaN(p.ATol) || math.IsInf(p.ATol, 0) {
		p.ATol = 1e-6 // default atol
	}

	if p.RTol == 0 || math.IsNaN(p.RTol) || math.IsInf(p.RTol, 0) {
		p.RTol = 1e-6 // default rtol
	}

	if p.Safety <= 0 || p.Safety > 1 ||
		math.IsNaN(p.Safety) || math.IsInf(p.Safety, 0) {
		p.Safety = 0.8 // default safety
	}

	return nil
}

type RKM struct {
	step solverStep
}

func (s *RKM) Init(p *Problem, prm *Parameters) {
	s.step = NewSolverStepRKM(p.YP, len(p.Y0), prm.ATol, prm.Safety)
}

func (s *RKM) Step(t float64, y []float64, dt float64) (float64, []float64, float64) {
	return s.step(t, y, dt)
}

// TODO find out what the name of this particular adaptor is.
type BasicAdaptor struct {
	safety float64
	tol    float64
}

func (a *BasicAdaptor) Init(p *Problem, prm *Parameters) {
	a.safety = prm.Safety
	a.tol = prm.ATol
}

func (a *BasicAdaptor) Adapt(dt, e float64) float64 {
	dt = a.safety * dt * math.Pow(a.tol/e, 0.2)
	return dt
}

type solverStep func(t float64, y []float64, dt float64) (float64, []float64, float64)

// Takes a function that calculates y' (y prime) and calculation parameters.
// Returns a function that performs a single step of the
// Runge-Kutta-Merson method with adaptive step.
func NewSolverStepRKM(yp YPrime, nEq int, tol, safety float64) solverStep {
	// we only want to allocate memory for increments once to avoid excessive gc
	k1 := make([]float64, nEq)
	k2 := make([]float64, nEq)
	k3 := make([]float64, nEq)
	k4 := make([]float64, nEq)
	k5 := make([]float64, nEq)
	// TODO but now len(y) might be different from nEq, check required
	solverStep :=
		func(t float64, y []float64, dt float64) (float64, []float64, float64) {
			copy(k1, y)
			k1 = yp(t, k1)

			copy(k2, y)
			for iEq := range y {
				k2[iEq] += dt * k1[iEq] / 3
			}
			k2 = yp(t+dt/3, k2)

			copy(k3, y)
			for iEq := range y {
				k3[iEq] += (k1[iEq] + k2[iEq]) * dt / 6
			}
			k3 = yp(t+dt/3, k3)

			copy(k4, y)
			for iEq := range y {
				k4[iEq] += (k1[iEq] + 3*k3[iEq]) * dt / 8
			}
			k4 = yp(t+dt/2, k4)

			copy(k5, y)
			for iEq := range y {
				k5[iEq] += (k1[iEq] - 3*k3[iEq] + 4*k4[iEq]) * dt / 2
			}
			k5 = yp(t+dt, k5)

			// Pick highest estimation of truncation error among all equations.
			e := 0.
			for iEq := range y {
				tmp := math.Abs((2*k1[iEq] - 9*k3[iEq] + 8*k4[iEq] - k5[iEq]) * dt / 30)
				e = math.Max(e, tmp)
			}

			// If error withing tolerance interval, proceed with the step.
			if e < tol {
				for iEq := range y {
					y[iEq] += (k1[iEq] + 4*k4[iEq] + k5[iEq]) * dt / 6
				}
				t += dt
			}

			return t, y, e
		}
	return solverStep
}

// Takes a function that calculates y' (y prime) and calculation parameters.
// Returns a function that performs a single step of the of the classical
// forth-order Runge-Kutta method.
func NewSolverStepRK4(yp YPrime, p Problem) solverStep {
	// we only want to allocate memory for increments once to avoid excessive gc
	dy1 := make([]float64, len(p.Y0))
	dy2 := make([]float64, len(p.Y0))
	dy3 := make([]float64, len(p.Y0))
	dy4 := make([]float64, len(p.Y0))
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
