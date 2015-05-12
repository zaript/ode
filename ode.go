// Copyright © 2015 Timur Zaripov. All Rights Reserved.
//
// Initial version of this code was written based on example from
// http://rosettacode.org/wiki/Runge-Kutta_method#Go
//
// Description of Runge-Kutta-Merson method can be found here:
// http://www.encyclopediaofmath.org/index.php/Kutta-Merson_method

// Package provides basic support for integration of ODE systems.
//
// Solves the initial value problem for systems of first order ode-s
//        dy/dt = f(y,t0,...)
// where y can be a vector.
//
// Current design:
// NewSolverStepXXX functions construct stepping functions from given
// function for right side of system (has YPrime signature) and solver
// parameters (passed as SolverPrm interface).
//
// Stepping functions perform single integration step for given time t,
// solution vector y and time step dt and return new values of t, y and
// possibly dt.
//
// Current issues:
// Parameters, that are required by yprime, are passed as part of a structure,
// that implements SolverPrm interface, and that doesn't look good.
// More tests required.
package ode

import m "math"

type SolverPrm interface {
	NEq() uint64     // Number of equations (variables).
	Tol() float64    // Estimated error tolerance for adaptive step control.
	Safety() float64 // Safety factor on new step selection
	TEnd() float64   // Finish time.
}

type YPrime func(t float64, y []float64, prm SolverPrm) []float64
type SolverStep func(t float64, y []float64, dt float64) (float64, []float64)
type AdaptiveSolverStep func(t float64, y []float64, dt float64) (float64, []float64, float64)

// Takes a function that calculates y' (y prime) and calculation
// parameters.
// Returns a function that performs a single step of the
// of the classical forth-order Runge-Kutta method.
func NewSolverStepRK4(yp YPrime, prm SolverPrm) SolverStep {
	// we only want to allocate memory for increments once to avoid excessive gc
	dy1 := make([]float64, prm.NEq())
	dy2 := make([]float64, prm.NEq())
	dy3 := make([]float64, prm.NEq())
	dy4 := make([]float64, prm.NEq())
	solverStep :=
		func(t float64, y []float64, dt float64) (float64, []float64) {
			copy(dy1, y)
			dy1 = yp(t, dy1, prm)

			copy(dy2, y)
			for iEq := range y {
				dy2[iEq] += dt * dy1[iEq] / 2
			}
			dy2 = yp(t+dt/2, dy2, prm)

			copy(dy3, y)
			for iEq := range y {
				dy3[iEq] += dt * dy2[iEq] / 2
			}
			dy3 = yp(t+dt/2, dy3, prm)

			copy(dy4, y)
			for iEq := range y {
				dy4[iEq] += dt * dy3[iEq]
			}
			dy4 = yp(t+dt, dy4, prm)

			for iEq := range y {
				y[iEq] += dt * (dy1[iEq] + 2*(dy2[iEq]+dy3[iEq]) + dy4[iEq]) / 6
			}
			return t + dt, y
		}
	return solverStep
}

// Takes a function that calculates y' (y prime) and calculation
// parameters.
// Returns a function that performs a single step of the
// Runge-Kutta-Merson method with adaptive step.
func NewSolverStepRKM(yp YPrime, prm SolverPrm) AdaptiveSolverStep {
	// we only want to allocate memory for increments once to avoid excessive gc
	dy1 := make([]float64, prm.NEq())
	dy2 := make([]float64, prm.NEq())
	dy3 := make([]float64, prm.NEq())
	dy4 := make([]float64, prm.NEq())
	dy5 := make([]float64, prm.NEq())
	solverStep :=
		func(t float64, y []float64, dt float64) (float64, []float64, float64) {
			copy(dy1, y)
			dy1 = yp(t, dy1, prm)

			copy(dy2, y)
			for iEq := range y {
				dy2[iEq] += dt * dy1[iEq] / 3
			}
			dy2 = yp(t+dt/3, dy2, prm)

			copy(dy3, y)
			for iEq := range y {
				dy3[iEq] += (dy1[iEq] + dy2[iEq]) * dt / 6
			}
			dy3 = yp(t+dt/3, dy3, prm)

			copy(dy4, y)
			for iEq := range y {
				dy4[iEq] += (dy1[iEq] + 3*dy3[iEq]) * dt / 8
			}
			dy4 = yp(t+dt/2, dy4, prm)

			copy(dy5, y)
			for iEq := range y {
				dy5[iEq] += (dy1[iEq] - 3*dy3[iEq] + 4*dy4[iEq]) * dt / 2
			}
			dy5 = yp(t+dt, dy5, prm)

			// Pick highest estimation of truncation error among all equations.
			err := 0.
			for iEq := range y {
				tmp := m.Abs((2*dy1[iEq] - 9*dy3[iEq] + 8*dy4[iEq] - dy5[iEq]) * dt / 30)
				err = m.Max(err, tmp)
			}

			// If error is in tolerance interval proceed with the step.
			if err < prm.Tol() {
				for iEq := range y {
					y[iEq] += (dy1[iEq] + 4*dy4[iEq] + dy5[iEq]) * dt / 6
				}
				t += dt
			}

			dt = m.Min(prm.Safety()*dt*m.Pow(prm.Tol()/err, 0.2), prm.TEnd()-t)

			return t, y, dt
		}
	return solverStep
}
