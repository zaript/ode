package ode

import (
	"fmt"
	"testing"
)

// Set of problems used for testing below composed by:
// Mazzia, Francesca and Magherini, Cecilia,
// Test Set for Initial Value Problem Solvers, release 2.4, February 2008,
// Department of Mathematics, University of Bari and INdAM, Research Unit of Bari,
// Available at http://www.dm.uniba.it/~testset.

// HIRES problem (ODE)
func Test_HIRES(test *testing.T) {

	hiresYPrime :=
		func(t float64, y []float64) []float64 {
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
	tRef := 321.8122
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

	hires := Problem{
		YP:     hiresYPrime,
		Y0:     []float64{1, 0, 0, 0, 0, 0, 0, 0.0057},
		T:      []float64{0, tRef},
		Dt0:    1.1e-3,
		ATol:   1.1e-6,
		RTol:   1.1e-6,
		Safety: 0.8,
	}

	solution, err := hires.Solve()
	if err != nil {
		test.Error("RKM: HIRES failed.")
	}

	if !allWithinTol(solution[1], yRef, hires.RTol, hires.ATol) {
		// TODO turn this into nice log output
		fmt.Println("y =", solution[1])
		fmt.Println("yRef =", yRef)
		test.Error("RKM: HIRES failed.")
	}
}
