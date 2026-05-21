module github.com/theapemachine/manifesto

go 1.26.1

replace github.com/theapemachine/hf => ../hf

replace github.com/theapemachine/manifesto => ../manifesto

require (
	github.com/smartystreets/goconvey v1.8.1
	github.com/theapemachine/hf v0.0.1
)

require (
	github.com/gopherjs/gopherjs v1.17.2 // indirect
	github.com/jtolds/gls v4.20.0+incompatible // indirect
	github.com/smarty/assertions v1.15.0 // indirect
	gopkg.in/yaml.v3 v3.0.1
)
