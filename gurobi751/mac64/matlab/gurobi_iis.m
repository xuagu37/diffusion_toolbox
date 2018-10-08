%GUROBI_IIS  Compute an Irreducible Inconsistent Subsytem (IIS)
%   IIS = GUROBI_IIS(MODEL, PARAMS) computes an IIS for the
%   optimization model stored in the MODEL struct.
%
%   An IIS is a subset of the constraints and variable bounds of the
%   original model. If all constraints in the model except those in an
%   IIS are removed, the model is still infeasible.  However, further
%   removing any one member of the IIS produces a feasible result.
%
%   The MODEL struct must contain a valid Gurobi model. See the
%   documentation of the GUROBI function for a description of this
%   struct's required fields and values.
%
%   The GUROBI_IIS function returns a struct IIS, with various results
%   stored in its named components. The specific results that are
%   available depend on the type of model.
%
%   The IIS struct will always contain the following fields
%
%   iis.minimal: A logical scalar that indicates whether the computed
%                IIS is minimal. It will normally be true, but it may
%                be false if the IIS computation was stopped early
%                (e.g due to a time limit or a user interrupt).
%
%   iis.constrs: A logical vector that indicates whether a linear
%                constraint appears in the computed IIS.
%
%   iis.lb: A logical vector that indicates whether a lower bound
%           appears in the computed IIS.
%
%   iis.ub: A logical vector that indicates whether a upper bound
%           appears in the computed IIS.
%
%   If your model contains SOS constraints the IIS struct will contain
%   the following field:
%
%  iis.sos: A logical vector that indicates whether an sos constraint
%           appears in the computed IIS
%
%   If your model contains cones or quadratic constraints the IIS
%   struct will contain the following field:
%
%   iis.qconstrs: A logical vector that indicates whether the cone or
%                 quadratic constraint appears in the computed IIS.
%                 Note that any cones in the model will appear
%                 first in this vector, followed by the quadratic
%                 constraints.
%
%  The PARAMS struct contains Gurobi parameters. A full list may be
%  found on the Parameter page of the reference manual:
%     http://gurobi.com/documentation/7.5/reference-manual/
%  The following parameters are relevant for computing an IIS.
%
%  params.resultfile: Specifies the name of the file to be written
%                     upon completion of computing an IIS. This file
%                     will contain the IIS. Filename must end with
%                     .ilp, .ilp.gz, or .ilp.bz2 suffix.
%
%  params.iismethod: Chooses the IIS method to use. Method 0 is often
%                    faster, while method 1 can produce a smaller
%                    IIS. The default value of -1 choose automaically.
%
% Example:
%
%    clear model params
%    model = gurobi_read('examples/data/klein1.mps')
%    params.resultfile = 'myiis.ilp';
%    iis   = gurobi_iis(model, params);
%
% Copyright 2016, Gurobi Optimization, Inc.
%
% See also GUROBI, GUROBI_READ, STRUCT
