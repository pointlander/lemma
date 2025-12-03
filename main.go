// Copyright 2025 The Lemma Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"math/cmplx"
	"math/rand"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

//go:embed iris.zip
var Iris embed.FS

// Fisher is the fisher iris data
type Fisher struct {
	Measures []float64
	Label    string
	Cluster  int
	Index    int
}

// Labels maps iris labels to ints
var Labels = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
}

// Inverse is the labels inverse map
var Inverse = [3]string{
	"Iris-setosa",
	"Iris-versicolor",
	"Iris-virginica",
}

// Load loads the iris data set
func Load() []Fisher {
	file, err := Iris.Open("iris.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	fisher := make([]Fisher, 0, 8)
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		panic(err)
	}
	for _, f := range reader.File {
		if f.Name == "iris.data" {
			iris, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(iris)
			data, err := reader.ReadAll()
			if err != nil {
				panic(err)
			}
			for i, item := range data {
				record := Fisher{
					Measures: make([]float64, 4),
					Label:    item[4],
					Index:    i,
				}
				for ii := range item[:4] {
					f, err := strconv.ParseFloat(item[ii], 64)
					if err != nil {
						panic(err)
					}
					record.Measures[ii] = f
				}
				fisher = append(fisher, record)
			}
			iris.Close()
		}
	}
	return fisher
}

// Random generates a random iris data set
func Random(seed int64) []Fisher {
	fisher, rng := make([]Fisher, 150), rand.New(rand.NewSource(seed))
	for i := range fisher {
		fisher[i].Measures = make([]float64, 4)
		for ii := range fisher[i].Measures {
			fisher[i].Measures[ii] = rng.Float64()
		}
		fisher[i].Label = fmt.Sprintf("%d", i)
		fisher[i].Index = i
	}
	return fisher
}

func main() {
	const (
		// S is the scaling factor for the softmax
		S = 1.0 - 1e-300
	)

	softmax := func(values []float64) {
		max := 0.0
		for _, v := range values {
			if v > max {
				max = v
			}
		}
		s := max * S
		sum := 0.0
		for j, value := range values {
			values[j] = math.Exp(value - s)
			sum += values[j]
		}
		for j, value := range values {
			values[j] = value / sum
		}
	}

	dot := func(a, b []float64) float64 {
		x := 0.0
		for i, value := range a {
			x += value * b[i]
		}
		return x
	}

	cs := func(a, b []float64) float64 {
		ab := dot(a, b)
		aa := dot(a, a)
		bb := dot(b, b)
		if aa <= 0 {
			return 0
		}
		if bb <= 0 {
			return 0
		}
		return ab / (math.Sqrt(aa) * math.Sqrt(bb))
	}

	process := func(iris []Fisher) float64 {
		data := make([]float64, 0, 4*len(iris))
		for _, value := range iris {
			data = append(data, value.Measures...)
		}
		// self attention
		a := mat.NewDense(len(iris), 4, data)
		adj := mat.NewDense(len(iris), len(iris), nil)
		adj.Mul(a, a.T())
		cp := mat.NewDense(len(iris), len(iris), nil)
		cp.Copy(adj)
		for r := range len(iris) {
			row := make([]float64, len(iris))
			for ii := range row {
				row[ii] = cp.At(r, ii)
			}
			softmax(row)
			cp.SetRow(r, row)
		}
		x := mat.NewDense(len(iris), 4, nil)
		x.Mul(cp, a)
		// eigenvector
		var eig mat.Eigen
		ok := eig.Factorize(adj, mat.EigenRight)
		if !ok {
			panic("Eigenvalue decomposition failed.")
		}
		eigenvectors := mat.NewCDense(len(iris), len(iris), nil)
		eig.VectorsTo(eigenvectors)
		i, j := make([]float64, 0, len(iris)), make([]float64, 0, len(iris))
		for r := range len(iris) {
			i = append(i, cmplx.Abs(eigenvectors.At(r, 0)))
			j = append(j, x.At(r, 0))
		}

		return cs(i, j)
	}

	iris, count := Load(), 0
	if process(iris) < .95 {
		count++
	}
	for i := range 128 {
		iris := Random(int64(i + 1))
		cs := process(iris)
		if cs < .95 {
			count++
		}
	}
	fmt.Printf("%d/129 outside of cosine similarity .95\n", count)
}
