# Copyright 2017, Gurobi Optimization, Inc.

PLATFORM = mac64
INC      = ../../include/
CC       = gcc
CPP      = g++
CARGS    = -m64 -g
CLIB     = -L../../lib/ -lgurobi75
RELEASE := $(shell uname -r | cut -f 1 -d .)
CPPLIB   = -L../../lib/ -lgurobi_c++ -lgurobi75
JSRC     = ../java
CLASSDIR = -classpath ../../lib/gurobi.jar:.
JFLAG    = -d . $(CLASSDIR)

all: genconstr_c poolsearch_c \
     callback_c dense_c diet_c facility_c feasopt_c fixanddive_c \
     lp_c lpmethod_c lpmod_c mip1_c mip2_c params_c piecewise_c qcp_c\
     qp_c sensitivity_c sos_c sudoku_c tsp_c tune_c multiobj_c \
     workforce1_c workforce2_c workforce3_c workforce4_c workforce5_c \
     callback_c++ dense_c++ diet_c++ facility_c++ feasopt_c++ fixanddive_c++ \
     lp_c++ lpmethod_c++ lpmod_c++ mip1_c++ mip2_c++ params_c++ piecewise_c++ \
     qcp_c++ qp_c++ sensitivity_c++ sos_c++ sudoku_c++ tsp_c++ tune_c++ \
     workforce1_c++ workforce2_c++ workforce3_c++ workforce4_c++ \
     workforce5_c++ genconstr_c++ poolsearch_c++ multiobj_c++ \
     Callback.class Dense.class Diet.class Facility.class Feasopt.class \
     Fixanddive.class Lp.class Lpmethod.class Lpmod.class Mip1.class \
     Mip2.class Params.class Piecewise.class Qcp.class Qp.class \
     Sensitivity.class Sos.class Sudoku.class Tsp.class Tune.class \
     Workforce1.class Workforce2.class Workforce3.class Workforce4.class \
     Workforce5.class Genconstr.class Poolsearch.class Multiobj.class

run: run_c run_c++ run_java run_python

run_c: run_callback_c run_dense_c run_diet_c run_facility_c run_feasopt_c \
       run_fixanddive_c run_lp_c run_lpmethod_c run_lpmod_c run_mip1_c \
       run_mip2_c run_params_c run_piecewise_c run_qcp_c run_qp_c \
       run_sensitivity_c run_sos_c run_sudoku_c run_tsp_c run_tune_c \
       run_workforce1_c run_workforce2_c run_workforce3_c run_workforce4_c \
       run_workforce5_c run_multiobj_c run_poolsearch_c run_genconstr_c

run_c++: run_callback_c++ run_dense_c++ run_diet_c++ run_facility_c++ \
         run_feasopt_c++ run_fixanddive_c++ run_lp_c++ run_lpmethod_c++ \
         run_lpmod_c++ run_mip1_c++ run_mip2_c++ run_params_c++ \
         run_piecewise_c++ run_qcp_c++ run_qp_c++ run_sensitivity_c++ \
         run_sos_c++ run_sudoku_c++ run_tsp_c++ run_tune_c++ \
         run_workforce1_c++ run_workforce2_c++ run_workforce3_c++ \
         run_workforce4_c++ run_workforce5_c++ run_multiobj_c++ \
         run_poolsearch_c++ run_genconstr_c++

run_java: Callback Dense Diet Facility Feasopt Fixanddive Lp Lpmethod Lpmod \
          Mip1 Mip2 Params Qcp Piecewise Qp Sensitivity Sos Sudoku Tsp Tune \
          Workforce1 Workforce2 Workforce3 Workforce4 Workforce5 \
          Genconstr Poolsearch Multiobj

run_python: run_callback_python run_dense_python run_diet_python \
        run_diet2_python run_diet3_python \
        run_facility_python run_feasopt_python run_fixanddive_python \
        run_lp_python run_lpmethod_python run_lpmod_python run_mip1_python \
        run_mip2_python run_netflow_python run_params_python \
        run_piecewise_python run_qcp_python run_qp_python \
        run_sensitivity_python run_sos_python run_sudoku_python \
        run_tsp_python run_tune_python run_workforce1_python \
        run_workforce2_python run_workforce3_python run_workforce4_python \
        run_workforce5_python run_genconstr_python \
        run_poolsearch_python run_multiobj_python


%_c: ../c/%_c.c
	$(CC) $(CARGS) -o $@ $< -I$(INC) $(CLIB) -lm

%_c++: ../c++/%_c++.cpp
	$(CPP) $(CARGS) -o $@ $< -I$(INC) $(CPPLIB) -lm

%.class: $(JSRC)/%.java
	javac $(JFLAG) $<

run_callback_c: callback_c
	./callback_c ../data/p0033.mps
run_dense_c: dense_c
	./dense_c
run_diet_c: diet_c
	./diet_c
run_facility_c: facility_c
	./facility_c
run_feasopt_c: feasopt_c
	./feasopt_c ../data/klein1.mps
run_fixanddive_c: fixanddive_c
	./fixanddive_c ../data/stein9.mps
run_lp_c: lp_c
	./lp_c ../data/klein1.mps
run_lpmethod_c: lpmethod_c
	./lpmethod_c ../data/afiro.mps
run_lpmod_c: lpmod_c
	./lpmod_c ../data/afiro.mps
run_mip1_c: mip1_c
	./mip1_c
run_mip2_c: mip2_c
	./mip2_c ../data/stein9.mps
run_params_c: params_c
	./params_c ../data/misc07.mps
run_piecewise_c: piecewise_c
	./piecewise_c
run_qcp_c: qcp_c
	./qcp_c
run_qp_c: qp_c
	./qp_c
run_sensitivity_c: sensitivity_c
	./sensitivity_c ../data/p0033.lp
run_sos_c: sos_c
	./sos_c
run_sudoku_c: sudoku_c
	./sudoku_c ../data/sudoku1
run_tsp_c: tsp_c
	./tsp_c 50
run_tune_c: tune_c
	./tune_c ../data/p0033.mps
run_workforce1_c: workforce1_c
	./workforce1_c
run_workforce2_c: workforce2_c
	./workforce2_c
run_workforce3_c: workforce3_c
	./workforce3_c
run_workforce4_c: workforce4_c
	./workforce4_c
run_workforce5_c: workforce5_c
	./workforce5_c
run_multiobj_c: multiobj_c
	./multiobj_c
run_poolsearch_c: poolsearch_c
	./poolsearch_c
run_genconstr_c: genconstr_c
	./genconstr_c

run_callback_c++: callback_c++
	./callback_c++ ../data/p0033.mps
run_dense_c++: dense_c++
	./dense_c++
run_diet_c++: diet_c++
	./diet_c++
run_facility_c++: facility_c++
	./facility_c++
run_feasopt_c++: feasopt_c++
	./feasopt_c++ ../data/klein1.mps
run_fixanddive_c++: fixanddive_c++
	./fixanddive_c++ ../data/stein9.mps
run_lp_c++: lp_c++
	./lp_c++ ../data/klein1.mps
run_lpmethod_c++: lpmethod_c++
	./lpmethod_c++ ../data/afiro.mps
run_lpmod_c++: lpmod_c++
	./lpmod_c++ ../data/afiro.mps
run_mip1_c++: mip1_c++
	./mip1_c++
run_mip2_c++: mip2_c++
	./mip2_c++ ../data/stein9.mps
run_params_c++: params_c++
	./params_c++ ../data/misc07.mps
run_piecewise_c++: piecewise_c++
	./piecewise_c++
run_qcp_c++: qcp_c++
	./qcp_c++
run_qp_c++: qp_c++
	./qp_c++
run_sensitivity_c++: sensitivity_c++
	./sensitivity_c++ ../data/p0033.lp
run_sos_c++: sos_c++
	./sos_c++
run_sudoku_c++: sudoku_c++
	./sudoku_c++ < ../data/sudoku1
run_tsp_c++: tsp_c++
	./tsp_c++ 50
run_tune_c++: tune_c++
	./tune_c++ ../data/p0033.mps
run_workforce1_c++: workforce1_c++
	./workforce1_c++
run_workforce2_c++: workforce2_c++
	./workforce2_c++
run_workforce3_c++: workforce3_c++
	./workforce3_c++
run_workforce4_c++: workforce4_c++
	./workforce4_c++
run_workforce5_c++: workforce5_c++
	./workforce5_c++
run_multiobj_c++: multiobj_c++
	./multiobj_c++
run_poolsearch_c++: poolsearch_c++
	./poolsearch_c++
run_genconstr_c++: genconstr_c++
	./genconstr_c++

Callback: Callback.class
	java $(CLASSDIR): Callback ../data/p0033.mps
Dense: Dense.class
	java $(CLASSDIR): Dense
Diet: Diet.class
	java $(CLASSDIR): Diet
Facility: Facility.class
	java $(CLASSDIR): Facility
Feasopt: Feasopt.class
	java $(CLASSDIR): Feasopt ../data/klein1.mps
Fixanddive: Fixanddive.class
	java $(CLASSDIR): Fixanddive ../data/stein9.mps
Lp: Lp.class
	java $(CLASSDIR): Lp ../data/klein1.mps
Lpmethod: Lpmethod.class
	java $(CLASSDIR): Lpmethod ../data/afiro.mps
Lpmod: Lpmod.class
	java $(CLASSDIR): Lpmod ../data/afiro.mps
Mip1: Mip1.class
	java $(CLASSDIR): Mip1
Mip2: Mip2.class
	java $(CLASSDIR): Mip2 ../data/stein9.mps
Params: Params.class
	java $(CLASSDIR): Params ../data/misc07.mps
Piecewise: Piecewise.class
	java $(CLASSDIR): Piecewise
Qcp: Qcp.class
	java $(CLASSDIR): Qcp
Qp: Qp.class
	java $(CLASSDIR): Qp
Sensitivity: Sensitivity.class
	java $(CLASSDIR): Sensitivity ../data/p0033.lp
Sos: Sos.class
	java $(CLASSDIR): Sos
Sudoku: Sudoku.class
	java $(CLASSDIR): Sudoku ../data/sudoku1
Tsp: Tsp.class
	java $(CLASSDIR): Tsp 50
Tune: Tune.class
	java $(CLASSDIR): Tune ../data/p0033.mps
Workforce1: Workforce1.class
	java $(CLASSDIR): Workforce1
Workforce2: Workforce2.class
	java $(CLASSDIR): Workforce2
Workforce3: Workforce3.class
	java $(CLASSDIR): Workforce3
Workforce4: Workforce4.class
	java $(CLASSDIR): Workforce4
Workforce5: Workforce5.class
	java $(CLASSDIR): Workforce5
Genconstr: Genconstr.class
	java $(CLASSDIR): Genconstr
Poolsearch: Poolsearch.class
	java $(CLASSDIR): Poolsearch
Multiobj: Multiobj.class
	java $(CLASSDIR): Multiobj

run_callback_python:
	gurobi.sh ../python/callback.py ../data/p0033.mps
run_dense_python:
	gurobi.sh ../python/dense.py
run_diet_python:
	gurobi.sh ../python/diet.py
run_diet2_python:
	gurobi.sh ../python/diet2.py
run_diet3_python:
	gurobi.sh ../python/diet3.py
run_diet4_python:
	gurobi.sh ../python/diet4.py
run_facility_python:
	gurobi.sh ../python/facility.py
run_feasopt_python:
	gurobi.sh ../python/feasopt.py ../data/klein1.mps
run_fixanddive_python:
	gurobi.sh ../python/fixanddive.py ../data/stein9.mps
run_lp_python:
	gurobi.sh ../python/lp.py ../data/klein1.mps
run_lpmethod_python:
	gurobi.sh ../python/lpmethod.py ../data/afiro.mps
run_lpmod_python:
	gurobi.sh ../python/lpmod.py ../data/afiro.mps
run_mip1_python:
	gurobi.sh ../python/mip1.py
run_mip2_python:
	gurobi.sh ../python/mip2.py ../data/stein9.mps
run_netflow_python:
	gurobi.sh ../python/netflow.py
run_params_python:
	gurobi.sh ../python/params.py ../data/misc07.mps
run_piecewise_python:
	gurobi.sh ../python/piecewise.py
run_qcp_python:
	gurobi.sh ../python/qcp.py
run_qp_python:
	gurobi.sh ../python/qp.py
run_sensitivity_python:
	gurobi.sh ../python/sensitivity.py ../data/p0033.lp
run_sos_python:
	gurobi.sh ../python/sos.py
run_sudoku_python:
	gurobi.sh ../python/sudoku.py ../data/sudoku1
run_tsp_python:
	gurobi.sh ../python/tsp.py 50
run_tune_python:
	gurobi.sh ../python/tune.py ../data/p0033.mps
run_workforce1_python:
	gurobi.sh ../python/workforce1.py
run_workforce2_python:
	gurobi.sh ../python/workforce2.py
run_workforce3_python:
	gurobi.sh ../python/workforce3.py
run_workforce4_python:
	gurobi.sh ../python/workforce4.py
run_workforce5_python:
	gurobi.sh ../python/workforce5.py
run_genconstr_python:
	gurobi.sh ../python/genconstr.py
run_poolsearch_python:
	gurobi.sh ../python/poolsearch.py
run_multiobj_python:
	gurobi.sh ../python/multiobj.py

clean:
	rm -rf *.o *_c *_c++ *.class *.log *.rlp *.lp *.bas *.ilp *.mps *.dSYM
