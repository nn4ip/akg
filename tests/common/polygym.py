import json, os
import textwrap, base64
import islpy as isl
import subprocess

# from math import gcd
# from functools import reduce
from fractions import Fraction, gcd
from functools import reduce

CHERNIKOVA_TIMEOUT = 60

def get_env_or_default(vairable_name, default_name):
    if vairable_name in os.environ:
        return os.environ[vairable_name]

    return default_name
polyite_dir = get_env_or_default('POLYITEDIR', '/devel/git_3rd/polyite')

class ChernikovaTimeoutException(Exception):
    pass

class Config:
    def __init__(self):
        self.numSchedsAtOnce = 1
        self.rayCoeffsRange = 1
        self.lineCoeffsRange = 1
        self.currNumRaysLimit = 2
        self.currNumLinesLimit = 2
        self.maxNumRays = 2
        self.maxNumLines = 2

class DomainCoeffInfo:
    def __init__(self, nrIt, nrParPS, stmtInfo, universe, domain):
        self.nrIt = nrIt
        self.nrParPS = nrParPS
        self.stmtInfo = stmtInfo
        self.universe = universe
        self.domain = domain
        self.nrStmts = len(stmtInfo)
        self.dim = nrIt + nrParPS * self.nrStmts + self.nrStmts
        self.scheduleParamSpace = domain.get_space().params()

    def __repr__(self):
        return "DomainCoeffInfo(nrIt = %r, nrParPS = %r, stmtInfo = %r, universe = %r, domain = %r)" % (
        self.nrIt, self.nrParPS, self.stmtInfo, self.universe, self.domain)


class StmtCoeffInfo:
    def __init__(self, itStart, nrIt, parStart, cstIdx):
        self.itStart = itStart
        self.nrIt = nrIt
        self.parStart = parStart
        self.cstIdx = cstIdx

    def __repr__(self):
        return "StmtCoeffInfo(itStart = %r, nrIt = %r, parStart = %r, cstIdx = %r)" % (
        self.itStart, self.nrIt, self.parStart, self.cstIdx)

class GeneratorInfo:
    def __init__(self, vertices, rays, lines):
        self.vertices = vertices
        self.rays = rays
        self.lines = lines

    def __repr__(self):
        return "GeneratorInfo(\n" + \
            wrap(self.vertices, "  vertices = ") + \
            wrap(self.rays, "  rays = ") + \
            wrap(self.lines, "  lines = ") + \
            ")"

class Dependence:
    def __init__(self, baseMap, weakConstr, strongConstr):
        self.baseMap = baseMap
        self.weakConstr = weakConstr
        self.strongConstr = strongConstr

        self.tupleNameIn = baseMap.get_tuple_name(isl.dim_type.in_)
        self.tupleNameOut = baseMap.get_tuple_name(isl.dim_type.out)

    def __repr__(self):
        return "Dependence(\n" + \
            wrap(self.baseMap, "  baseMap = ") + \
            wrap(self.weakConstr, "  weakConstr = ") + \
            wrap(self.strongConstr, "  strongConstr = ") + \
            ")"

class Access:
    def __init__(self, kind, relation):
        self.kind = kind
        self.relation = relation

    def toJsonDict(self):
        return {
            "kind": self.kind,
            "relation": self.relation.to_str()
        }

    def __repr__(self):
        return "Access(kind = %r, relation = %r)" % (self.kind, self.relation)

class Statement:
    def __init__(self, domain, name, schedule, accesses):
        self.domain = domain
        self.name = name
        self.schedule = schedule
        self.accesses = accesses

    def toJsonDict(self):
        return {
            "domain": self.domain.to_str(),
            "name": self.name,
            "schedule": self.schedule.to_str(),
            "accesses": [x.toJsonDict() for x in self.accesses]
        }

    def __repr__(self):
        return "Statement(name = %r, domain = %r, schedule = %r, accesses = %r)" % (
        self.name, self.domain, self.schedule, self.accesses)

class Scop:
    def __init__(self, domain, schedule, context, statements, params, scheduleTree=None, memoryTransforms=None):
        self.domain = domain
        self.schedule = schedule
        self.context = context
        self.statements = statements
        self.params = params
        self.scheduleTree = scheduleTree
        self.memoryTransforms = {} if memoryTransforms is None else memoryTransforms
        self.reads_ = None
        self.writes_ = None

    def toJsonDict(self):
        return {
            'statements': [x.toJsonDict() for x in self.statements],
            'memoryTransforms': {k: v.to_str() for k, v in self.memoryTransforms.items()}
        }

    def reads(self):
        if self.reads_ is None:
            self.reads_ = self.populateMemAccess_("read")
        return self.reads_

    def writes(self):
        if self.writes_ is None:
            self.writes_ = self.populateMemAccess_("write")
        return self.writes_

    def populateMemAccess_(self, memType):
        result = isl.UnionMap.empty(self.context.get_space())
        for stmt in self.statements:
            for access in stmt.accesses:
                if access.kind == memType:
                    domain = stmt.domain
                    result = result.add_map(access.relation.intersect_domain(domain))
        return result.coalesce()

    def splitStatement(self, name, partialDomain):
        stmt = None
        for candidate in self.statements:
            if candidate.name == name:
                stmt = candidate
                break
        if stmt is not None:
            oldDomain = stmt.domain
            oldSchedule = stmt.schedule
            newStmt = copy.deepcopy(stmt)
            stmt.domain = stmt.domain.subtract(partialDomain)
            stmt.name = stmt.name + "a"
            # Do not change the intersection order, as this keeps the dim names of the old domain
            # These need to be kept, otherwise the mem references do not match any more
            newStmt.domain = oldDomain.intersect(partialDomain)
            newStmt.name = newStmt.name + "b"
            self.domain = self.domain.subtract(oldDomain)
            self.schedule = self.schedule.subtract(oldSchedule)
            for s in [stmt, newStmt]:
                s.domain = s.domain.set_tuple_name(s.name)
                s.schedule = s.schedule.set_tuple_name(isl.dim_type.in_, s.name)
                self.domain = self.domain.add_set(s.domain)
                self.schedule = self.schedule.add_map(s.schedule)
                for access in s.accesses:
                    access.relation = access.relation.set_tuple_name(isl.dim_type.in_, s.name)
            self.statements.append(newStmt)
            self.writes_ = None
            self.reads_ = None

    def __repr__(self):
        return "Scop(domain = %r, schedule = %r, context = %r, statements = %r)" % (
        self.domain, self.schedule, self.context, self.statements)

class Schedule():
    def __init__(self, domInfo, deps):
        self.domInfo = domInfo
        self.deps = deps

        self.scheduleSummands = []
        self.scheduleVectors = []
        self.dim2StronglySatisfiedDeps = []

    def __getDirectionOfDep(self, sched, dep):
        """
        Determines the direction that schedule {@code sched} imposes on
        dependence {@code dep}.
        """
        newDep = dep.baseMap.apply_domain(sched).apply_range(sched)
        sp = sched.get_space()
        sp = sp.add_dims(isl.dim_type.in_, 1).add_dims(isl.dim_type.out, 1)
        delta = newDep.extract_map(sp).deltas()

        ctx = dep.baseMap.get_ctx()
        univ = isl.Set.universe(delta.get_space())
        zer = univ.fix_val(isl.dim_type.set, 0, isl.Val.zero(ctx))
        pos = univ.lower_bound_val(isl.dim_type.set, 0, isl.Val.one(ctx))
        neg = univ.upper_bound_val(isl.dim_type.set, 0, isl.Val.negone(ctx))

        isNeg = not delta.intersect(neg).is_empty()
        isZer = not delta.intersect(zer).is_empty()
        isPos = not delta.intersect(pos).is_empty()

        if isNeg and not isZer and not isPos:
            return -1
        if isZer and not isNeg and not isPos:
            return 0
        if isPos and not isZer and not isNeg:
            return 1

    def __getDepsStronglySatisfiedBySchedule(self, coeffs, deps, domInfo):
        """
        Filters {@code deps} for dependencies that are satisfied strongly by the
        schedule given through the coefficient matrix {@code coeffs}.
        """
        ret = []
        for dep in deps:
            schedMap = coeffMatrix2IslUnionMap(domInfo, coeffs)
            direction = self.__getDirectionOfDep(schedMap, dep)
            if direction == 1:
                ret.append(dep)
        return ret

    def __updateStronglySatisfiedDependences(self, dim):
        """
        Bookkeeping of the dependences that are satisfied strongly by each schedule
        dimension.
        """
        schedCoeffs = multiplyWithCommonDenominator(self.scheduleVectors[dim])
        carriedDeps = self.__getDepsStronglySatisfiedBySchedule(schedCoeffs, self.deps, self.domInfo)
        if len(self.dim2StronglySatisfiedDeps) == dim:
            self.dim2StronglySatisfiedDeps.append(carriedDeps)
        else:
            self.dim2StronglySatisfiedDeps[dim] = carriedDeps

    def getNumDims(self):
        return len(self.scheduleVectors)

    # def isConstant(self, dim):
    #     """
    #     Check whether dimension {@code dim} is a constant schedule dimension.
    #     """
    #     coeffs = self.scheduleVectors[dim]
    #     return checkIsConstant(coeffs, self.domInfo)

    def getDependencesCarriedUpToDim(self, dim):
        """
        Returns the set of dependences that are carried strongly up to dimension
        :param dim: (Starting from the outermost dimension).
        """
        ret = []
        for i in range(0, dim + 1):
            ret += self.dim2StronglySatisfiedDeps[i]

        return ret

    def getCarriedDeps(self):
        """
        Returns the set of all dependences that are carried by this
        schedule.
        """
        if self.getNumDims() > 0:
            return self.getDependencesCarriedUpToDim(self.getNumDims() - 1)
        return []

    def addScheduleVector(self, coeffs, schedSummands, dependencies=None):
        """
        Appends a new (inner) dimension to this schedule.
        (With dependencies parameter: This avoids costly recalculating of dependencies,
        if dependencies are are already known. Used when migrating schedules between MPI nodes.)

        :param coeffs: coeffs rational schedule coefficients of the new schedule dimension.
        :param schedSummands: schedSummands the linear combination of Chernikova generators that formed {@code coeffs}.
        :param dependencies: the dependencies strongly satisfied by this dimension
        :return:
        """
        self.scheduleSummands.append(schedSummands)
        self.scheduleVectors.append(coeffs)
        if dependencies:
            self.dim2StronglySatisfiedDeps.append(dependencies)
        else:
            self.__updateStronglySatisfiedDependences(len(self.scheduleVectors) - 1)

    def getScheduleVector(self, dim):
        """
        Get the schedule coefficients of dimension {@code dim}.
        """
        return self.scheduleVectors[dim]

    def addForeignDim(self, o, otherDim):
        """
        Appends a new (inner) dimension to this schedule that is equal to
        dimension {@code otherDim} of schedule {@code o}.
        """
        if otherDim >= o.getNumDims():
            raise Exception("ownDim must not be >= o.numDims (" + str(o.getNumDims()) + "): " + str(otherDim))
        self.scheduleSummands.append(o.scheduleSummands[otherDim])
        self.scheduleVectors.append(o.scheduleVectors[otherDim])
        self.__updateStronglySatisfiedDependences(len(self.scheduleVectors) - 1)

    def computeLinDepSpaceInt(self, domInfo, vectors, start, nr):
        def t2Val(self, i):
            return isl.Val.int_from_si(domInfo.xtx, i)

        if len(vectors) == 0:
            return None

        ctx = domInfo.ctx
        nrPoints = len(vectors)

        mAff = isl.MultiAff.zero(isl.Space.alloc(ctx, 0, nrPoints, nr))
        for pos in range(0, nr + 1):
            aff = isl.Aff.zero_on_domain(isl.LocalSpace.from_space(isl.Space.set_alloc(ctx, 0, nrPoints)))
            for i, cs in enumerate(vectors):
                aff = aff.set_coefficient_val(isl.dim_type.in_, i, t2Val(cs[start + pos]))
            mAff = mAff.set_aff(pos, aff)
        linDepSpace = isl.BasicMap.from_multi_aff(mAff).range()
        # remove existentially qualified variables in linDepSace (with removeDivs)
        # because there could be holes in the space... [1 0 | 0 2] does not span the full 2D space
        linDepSpace = linDepSpace.remove_divs()

        # expand to full dimensionality
        return linDepSpace.insert_dims(isl.dim_type.set, 0, start).add_dims(isl.dim_type.set, domInfo.dim - (start + nr))

    # def computeLinIndepSpaceInt(self, domInfo, vectors, start, nr):
    #     if len(vectors) == 0:
    #         return None
    #     return computeLinDepSpace(domInfo, vectors, start, nr).complement()

    # def computeLinIndepSpace(self, maxDim, fixFreeDims):
    #     """
    #     Compute the set of schedule coefficient vectors that is linearly
    #     independent to the schedule coefficient vectors of this schedule from
    #     dimension 0 (inclusive) to dimension {@code maxDim} (exclusive). Linear
    #     independence is only required for the coefficients of iteration variables.

    #     @return Returns {@code None} if the resulting space is unconstrained.
    #     Otherwise returns {@code Some(sp)}.
    #     """
    #     if maxDim == 0 and self.getNumDims() == 0 and self.domInfo.nrIt > 0:
    #         return self.domInfo.universe
    #     if maxDim < 0 or maxDim > self.getNumDims():
    #         raise Exception("maxDim must be from the interval [0, " + str(self.getNumDims()) + "(: " + str(maxDim))

    #     schedVects = [multiplyWithCommonDenominator(x) for x in self.scheduleVectors[0:maxDim]]

    #     if len(schedVects) == 0:
    #         return None

    #     result = self.domInfo.universe

    #     zeroVal = isl.Val.zero(self.domInfo.ctx)
    #     oneVal = isl.Val.one(self.domInfo.ctx)
    #     allZero = true

    #     for sci in self.domInfo.stmtInfo:
    #         # linear independence for iterator coeffs only is required (params and consts are irrelevant)
    #         linIndepSpace = computeLinIndepSpaceBigInt(self.domInfo, self.schedVects, sci.itStart, sci.nrIt)
    #         if linIndepSpace == None or isl.simplify(linIndepSpace).is_empty():
    #             if fixFreeDims:
    #                 # dimension for this statement irrelevant... no exploration required, set anything
    #                 if sci.nrIt > 0:
    #                     result = result.fixVal(isl.dim_type.set, sci.itStart, oneVal)
    #                 for i in range(sci.itStart + 1, sci.itStart + sci.nrIt + 1):
    #                     result = result.fixVal(isl.dim_type.set, i, zeroVal)
    #                 for i in range(sci.parStart, sci.parStart + self.domInfo.nrParPS):
    #                     result = result.fixVal(isl.dim_type.set, i, zeroVal + 1)
    #                 result = result.fixVal(isl.dim_type.set, sci.cstIdx, zeroVal)

    #         else:
    #             allZero = False
    #             result = result.intersect(linIndepSpace)

    #     if allZero:
    #         return None
    #     else:
    #         return result

    def __repr__(self):
        schedSummandsStr = ''
        for dim, schedSummandsPerDim in enumerate(self.scheduleSummands):
            schedSummandsStr += 'Dimension: %i\n' % dim
            for schedSummand in schedSummandsPerDim:
                if schedSummand.coeff == 0:
                    continue
                schedSummandsStr += '  %s\n' % schedSummand

        return "Schedule(\n" + \
            "  schedSummands =\n" + schedSummandsStr + \
            ")"

# From ScheduleSummand
class ScheduleSummand():
    """
    Single monomial of the linear combination that forms a schedule vectors.
    """
    def __init__(self, v, coeff):
        self.v = v
        self.coeff = coeff

    def __eq__(self, other):
        if isinstance(other, ScheduleSummand):
            return other.coeff == self.coeff and len(other.v) == len(self.v) and other.v == self.v
        return False

    def getValue(self):
        return [x * self.coeff for x in self.v]

class VertexSummand(ScheduleSummand):
    def __eq__(self, other):
        return isinstance(other, VertexSummand) and super.__eq__(other)

    def __repr__(self):
        return 'VertexSummand:\t%i * %s' % (self.coeff, str(self.v))


class RaySummand(ScheduleSummand):
    def __eq__(self, other):
        return isinstance(other, RaySummand) and super.__eq__(other)

    def __repr__(self):
        return 'RaySummand:\t\t%i * %s' % (self.coeff, str(self.v))


class LineSummand(ScheduleSummand):
    def __eq__(self, other):
        return isinstance(other, LineSummand) and super.__eq__(other)

    def __repr__(self):
        return 'LineSummand:\t\t%i * %s' % (self.coeff, str(self.v))

def computeDomInfo(domain):
    ctx = domain.get_ctx()
    count = 0
    i = 0
    nrStmts = domain.n_set()
    domainSets = list()
    setList = domain.get_set_list()
    for i in range(setList.n_set()):
        lSet = setList.get_set(i)
        domainSets.append(lSet)
        count = count + lSet.dim(isl.dim_type.set)
        i += 1

    def exKey(lSet):
        return lSet.get_tuple_name()

    domainSets.sort(key=exKey)
    stmtInfo = dict()
    nrIt = count
    count = 0
    domainParDim = domain.params().dim(isl.dim_type.param)
    i = 0
    for lSet in domainSets:
        stmtNrIt = lSet.dim(isl.dim_type.set)
        stmtInfo[lSet.get_tuple_name()] = StmtCoeffInfo(count, stmtNrIt, nrIt + domainParDim * i,
                                                        nrIt + domainParDim * nrStmts + i)
        count = count + stmtNrIt
        i = i + 1
    dim = nrIt + nrStmts * (domainParDim + 1)
    universe = isl.Set.universe(isl.Space.set_alloc(ctx, 0, dim))
    return DomainCoeffInfo(nrIt, domainParDim, stmtInfo, universe, domain)

def wrap(content, prefix=None):
    wrapper = textwrap.TextWrapper(initial_indent=prefix if prefix else '', width=120,
                                   subsequent_indent=' ' * len(prefix) if prefix else '')
    return wrapper.fill(str(content)) + "\n"

def preprocess(deps, domain):
    """Only dependencies and domain are needed as parameters"""
    result = list()
    maxSplit = 100

    def depIterator(lMap):
        intermediate = list()
        i = 0
        mapUnwrapped = lMap.uncurry().domain().unwrap() if lMap.range_is_wrapping() else lMap
        depsMap = mapUnwrapped
        # simulate do-while loop
        while True:
            dep = depsMap.lexmin().coalesce()
            basicMapList = dep.get_basic_maps()
            # convert lambda foreach to python loop
            for basicMap in basicMapList:
                intermediate.append(basicMap.remove_divs())
                i = i + 1
                if i > maxSplit:
                    break
            depsMap = depsMap.subtract(dep)
            # invert condition
            if (depsMap.is_empty() or i > maxSplit):
                break
        # end of while loop
        if (i <= maxSplit):
            result.extend(intermediate)
        else:
            basicMapList = mapUnwrapped.get_basic_maps()
            for basicMap in basicMapList:
                result.append(basicMap.remove_divs())

    # end depIterator
    deps.foreach_map(depIterator)
    result = [x for x in result if not x.intersect_domain(domain).intersect_range(domain).is_empty()]

    def sortKey(basicMap):
        return max(basicMap.get_tuple_name(isl.dim_type.in_), basicMap.get_tuple_name(isl.dim_type.out))

    result.sort(key=sortKey)
    return result

def compSchedConstrForDep(dep, domInfo, strongSatisfy):
    ctx = dep.get_ctx()
    depSrcName = dep.get_tuple_name(isl.dim_type.in_)
    depDstName = dep.get_tuple_name(isl.dim_type.out)
    inEqOut = depSrcName == depDstName

    srcDim = dep.dim(isl.dim_type.in_)
    dstDim = dep.dim(isl.dim_type.out)
    parDim = dep.dim(isl.dim_type.param)
    depPoly = dep.move_dims(isl.dim_type.in_, srcDim, isl.dim_type.out, 0, dstDim).domain()
    schedCoeffs = depPoly.coefficients().unwrap()

    # see if this is really necessary
    if schedCoeffs.has_tuple_id(isl.dim_type.out):
        # convert to basic map
        schedCoeffs = schedCoeffs.reset_tuple_id(isl.dim_type.out).get_basic_map_list.get_map(0)

    njuDim = srcDim if inEqOut else srcDim + dstDim
    postprocIt = isl.Map.universe(isl.Space.alloc(ctx, 0, srcDim + dstDim, njuDim))
    for i in range(srcDim):
        postprocIt = postprocIt.oppose(isl.dim_type.in_, i, isl.dim_type.out, i)
    njuOff = 0 if inEqOut else srcDim
    for i in range(dstDim):
        postprocIt = postprocIt.equate(isl.dim_type.in_, srcDim + i, isl.dim_type.out, njuOff + i)
    schedCoeffs = schedCoeffs.apply_range(postprocIt.affine_hull())

    # print("")
    # print("For constraint: %r" % (dep,))
    # print("Sched coefficients in the middle: %r" % (schedCoeffs,))

    postprocPar = None
    if inEqOut:
        postprocPar = isl.BasicMap.universe(isl.Space.alloc(ctx, 0, parDim + 1, 0))
        zeroVal = isl.Val.zero(ctx)
        valC = isl.Val.negone(ctx) if strongSatisfy else zeroVal
        postprocPar = postprocPar.fix_val(isl.dim_type.in_, 0, valC)
        for i in range(parDim):
            postprocPar = postprocPar.fix_val(isl.dim_type.in_, i + 1, zeroVal)
    else:
        mAff = isl.MultiAff.zero(isl.Space.alloc(ctx, parDim + 1, parDim + 1, 2 * parDim + 2))
        lSp = isl.LocalSpace.from_space(isl.Space.set_alloc(ctx, parDim + 1, parDim + 1))
        oneVal = isl.Val.one(ctx)
        affC1 = isl.Aff.var_on_domain(lSp, isl.dim_type.param, 0)
        affC2 = isl.Aff.var_on_domain(lSp, isl.dim_type.set, 0).add(affC1)
        if strongSatisfy:
            affC2 = isl.Aff.val_on_domain(lSp, oneVal).add(affC2)
        mAff = mAff.set_aff(0, affC1)
        mAff = mAff.set_aff(1, affC2)
        # print("")
        # print("For constraint: %r" % (dep,))
        # print("mAff: %r" % (mAff,))
        for i in range(parDim):
            affP1 = isl.Aff.var_on_domain(lSp, isl.dim_type.param, i + 1)
            affP2 = isl.Aff.var_on_domain(lSp, isl.dim_type.set, i + 1).add(affP1)
            mAff = mAff.set_aff(i + 2, affP1)
            mAff = mAff.set_aff(i + 2 + parDim, affP2)
        postprocPar = isl.BasicMap.from_multi_aff(mAff).project_out(isl.dim_type.param, 0, parDim + 1)
    # endif

    schedCoeffs = schedCoeffs.apply_domain(postprocPar)

    # print("")
    # print("For constraint: %r" % (dep,))
    # print("Postproc par: %r", (postprocPar,))
    # print("Sched coefficients after apply domain: %r" % (schedCoeffs,))

    # expand dimensionality to global schedule constraints
    stmtSrcInfo = domInfo.stmtInfo[depSrcName]
    stmtDstInfo = domInfo.stmtInfo[depDstName]
    (srcItOff, srcNrIt, srcParOff, srcCstIdx) = (
    stmtSrcInfo.itStart, stmtSrcInfo.nrIt, stmtSrcInfo.parStart, stmtSrcInfo.cstIdx)
    (dstItOff, dstNrIt, dstParOff, dstCstIdx) = (
    stmtDstInfo.itStart, stmtDstInfo.nrIt, stmtDstInfo.parStart, stmtDstInfo.cstIdx)

    nrPar = domInfo.nrParPS
    solSpMap = isl.BasicMap.from_domain_and_range(schedCoeffs.reverse().wrap().flatten(),
                                                  isl.BasicSet.universe(isl.Space.set_alloc(ctx, 0, domInfo.dim)))

    off = 0
    for i in range(srcNrIt):
        solSpMap = solSpMap.equate(isl.dim_type.in_, off + i, isl.dim_type.out, srcItOff + i)
    if not inEqOut:
        off = off + srcNrIt
        for i in range(dstNrIt):
            solSpMap = solSpMap.equate(isl.dim_type.in_, off + i, isl.dim_type.out, dstItOff + i)
        off = off + dstNrIt

        solSpMap = solSpMap.equate(isl.dim_type.in_, off, isl.dim_type.out, srcCstIdx)
        off = off + 1
        solSpMap = solSpMap.equate(isl.dim_type.in_, off, isl.dim_type.out, dstCstIdx)
        off = off + 1

        for i in range(nrPar):
            solSpMap = solSpMap.equate(isl.dim_type.in_, off + i, isl.dim_type.out, srcParOff + i)
        off = off + nrPar
        for i in range(nrPar):
            solSpMap = solSpMap.equate(isl.dim_type.in_, off + i, isl.dim_type.out, dstParOff + i)

    return solSpMap.range()

def bMaps2Deps(bMaps, domInfo, islComputeout):
    result = list()
    for bMap in bMaps:
        ctx = bMap.get_ctx()
        oldMaxOps = ctx.get_max_operations()
        ctx.set_max_operations(islComputeout)
        # simplify for basic map is not coalesce, but this:
        dSimpl = bMap.remove_redundancies()
        weakConstr = compSchedConstrForDep(dSimpl, domInfo, False)
        strongConstr = compSchedConstrForDep(dSimpl, domInfo, True)
        ctx.reset_operations()
        ctx.set_max_operations(oldMaxOps)
        result.append(Dependence(bMap, weakConstr, strongConstr))
    return result


def printFlow(must, may, must_no, may_no):
    print("Must dep: %r" % (must,))
    print("May dep: %r" % (may,))
    print("Must No Src: %r" % (must_no,))
    print("May No Src: %r" % (may_no,))

def calcDepsAndDomInfo(params, domain, sched, reads, writes, additional=None):
    empty = isl.UnionMap.empty(domain.get_space().params())
    schedule = sched.intersect_domain(domain).coalesce()

    writeFlow = isl.UnionAccessInfo.from_sink(writes)
    writeFlow = writeFlow.set_must_source(writes)
    writeFlow = writeFlow.set_may_source(reads)
    writeFlow = writeFlow.set_schedule_map(schedule)
    res = writeFlow.compute_flow()
    may_dep = res.get_may_dependence()
    # (_,must_dep, may_dep, must_no_source, may_no_source) = writes.compute_flow(writes, reads, schedule)
    antiOut = may_dep
    # printFlow(must_dep, may_dep, must_no_source, may_no_source)
    readFlow = isl.UnionAccessInfo.from_sink(reads)
    readFlow = readFlow.set_must_source(writes)
    readFlow = readFlow.set_may_source(empty)
    readFlow = readFlow.set_schedule_map(schedule)
    res = readFlow.compute_flow()
    must_dep = res.get_must_dependence()

    # (_,must_dep, may_dep, must_no_source, may_no_source) = reads.compute_flow(writes, empty, schedule)
    # printFlow(must_dep, may_dep, must_no_source, may_no_source)
    flow = must_dep
    deps = antiOut.union(flow).coalesce()
    if additional is not None:
        deps = deps.union(additional).coalesce()
    # print(deps)
    domInfo = computeDomInfo(domain)
    depList = preprocess(deps, domain)
    # print("-----")
    # print("Dep list after preprocess:")
    # print(depList)
    dependences = bMaps2Deps(depList, domInfo, 0)

    return (dependences, domInfo)

def createScopFromDeserializedJson(dJson):
    context = isl.Set(dJson["context"])
    scheduleTree = None if "scheduleTree" not in dJson else isl.Schedule(dJson["scheduleTree"])
    statements = []
    for dStatement in dJson["statements"]:
        domain = isl.Set(dStatement["domain"])
        name = dStatement["name"]
        schedule = isl.Map(dStatement["schedule"])
        accesses = []
        for dAccess in dStatement["accesses"]:
            accesses.append(Access(dAccess["kind"], isl.Map(dAccess["relation"])))
        statements.append(Statement(domain, name, schedule, accesses))
    params = isl.UnionSet(dJson["context"])

    # calculate union_domains and union_schedule
    schedule = isl.UnionMap.empty(context.get_space())
    domain = isl.UnionSet.empty(context.get_space())
    for statement in statements:
        schedule = schedule.union(statement.schedule)
        domain = domain.union(statement.domain)
    memoryTransforms = {k: isl.Map(v) for k, v in
                        dJson['memoryTransforms'].items()} if 'memoryTransforms' in dJson else None
    return Scop(domain, schedule, context, statements, params, scheduleTree, memoryTransforms)

def coeffMatrix2IslUnionMap(domInfo, coeffs):
    schedule = isl.UnionMap.empty(domInfo.scheduleParamSpace)

    if type(coeffs[0]) is list:
        coeffsA = coeffs
    else:
        coeffsA = [coeffs]

    for stmt, sInfo in domInfo.stmtInfo.items():
        setSpace = domInfo.scheduleParamSpace.add_dims(isl.dim_type.set, sInfo.nrIt)
        madSpace = domInfo.scheduleParamSpace.add_dims(isl.dim_type.in_, sInfo.nrIt).add_dims(isl.dim_type.out, len(coeffs))
        mAff = isl.MultiAff.zero(madSpace)
        lspace = isl.LocalSpace.from_space(setSpace)

        for dim, coeff in enumerate(coeffsA):
            aff = isl.Aff.zero_on_domain(lspace)
            for i in range(0, sInfo.nrIt):
                aff = aff.set_coefficient_val(isl.dim_type.in_, i, coeff[sInfo.itStart + i])

            for i in range(0, domInfo.nrParPS):
                aff = aff.set_coefficient_val(isl.dim_type.param, i, coeff[sInfo.parStart + i])
            aff = aff.set_constant_val(coeff[sInfo.cstIdx])
            mAff = mAff.set_aff(dim, aff)
        stmtSchedule = isl.Map.from_multi_aff(mAff).set_tuple_name(isl.dim_type.in_, stmt)
        schedule = schedule.union(stmtSchedule)
    return schedule

def preparePolyhedron(constraintRepr, timeout=CHERNIKOVA_TIMEOUT):
    cmd = ["/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java",
           "-Djava.library.path=" + polyite_dir + "/polyite/scala-isl-utils/lib",
           "-classpath",
           polyite_dir + "/polyite/chernikova/target/scala-2.11/chernikova_2.11-0.1.0-SNAPSHOT.jar:" + polyite_dir + "/polyite/scala-isl-utils/target/scala-2.11/isl_2.11-0.1.0-SNAPSHOT.jar:" + polyite_dir + "/polyite/scala-isl-utils/lib/isl-scala.jar:" + polyite_dir + "/polyite/lib/scala-library.jar:" + polyite_dir + "/polyite/scala-isl-utils/lib",
           "org.exastencils.schedopt.chernikova.Main",
           str(constraintRepr)]
    if timeout:
        cmd = ["timeout", "%f" % timeout] + cmd

    try:
        out = subprocess.check_output(cmd)
    except subprocess.CalledProcessError as grepexc:
        if grepexc.returncode != 124:           # This is the error code for timed out execution
            print("error code", grepexc.returncode, grepexc.output)
        raise ChernikovaTimeoutException

    out = out.decode('utf-8')

    def parsingHelper(prefix, line):
        # Make JSON format compatible
        for old, new in [('(', '['), (')', ']'), ("{", "["), ("}", "]"), (";", ",")]:
            line = line.replace(old, new)
        # Remove line prefix
        line = line[len(prefix):]

        # line_parsed = json.loads(line)
        line = line.split('], [')
        line = [x.replace('[ [', '').replace('] ]', '') for x in line]
        line_decimal = []
        for l in line:
            l_decimal = []
            for n in l.split(','):
                l_decimal.append(eval(n + '.0' if '.' not in n else n))
            line_decimal.append(l_decimal)
        line_parsed = line_decimal

        return line_parsed

    ret = {
        'vertices': [],
        'rays': [],
        'lines': [],
    }
    for line in out.splitlines():
        for prefix in ['Vertices = ', 'Rays = ', 'Lines = ']:
            if line.startswith(prefix):
                ret[prefix.split()[0].lower()].append(parsingHelper(prefix, line))

    return GeneratorInfo(
        sorted(ret['vertices'][0], reverse=True),
        sorted(ret['rays'][0], reverse=True),
        sorted(ret['lines'][0], reverse=True))


def transformGeneratorInfoLinesToRays(generatorInfo):
    lines_as_rays = []
    for line in generatorInfo.lines:
        lines_as_rays.append(line)
        lines_as_rays.append([x * -1 for x in line])

    return GeneratorInfo(
        generatorInfo.vertices,
        sorted(generatorInfo.rays + lines_as_rays, reverse=True),
        [])

def islScheduleMap2IslScheduleTree(jsonp, isl_schedule_map):
    str_scop = base64.b64encode(json.dumps(jsonp).encode('ascii'))
    str_isl_schedule = base64.b64encode(str(isl_schedule_map).encode('ascii'))

    print(str_scop, str_isl_schedule)

    cmd = ["/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java",
           "-Djava.library.path=" + polyite_dir + "/polyite/scala-isl-utils/lib",
           "-classpath",
           polyite_dir + "/polyite/target/scala-2.11/polyite_2.11-0.1.0-SNAPSHOT.jar:" + polyite_dir + "/polyite/chernikova/target/scala-2.11/chernikova_2.11-0.1.0-SNAPSHOT.jar:" + polyite_dir + "/polyite/scala-isl-utils/target/scala-2.11/isl_2.11-0.1.0-SNAPSHOT.jar:" + polyite_dir + "/polyite/lib/scala-library.jar:" + polyite_dir + "/polyite/scala-isl-utils/lib/isl-scala.jar:" + polyite_dir + "/polyite/lib/mpi.jar:" + polyite_dir + "/polyite/lib/commons-lang3-3.4.jar:" + polyite_dir + "/polyite/lib/scala-parser-combinators_2.11-1.0.3.jar",
           "polyite.IslScheduleMap2IslScheduleTree",
           str_scop, str_isl_schedule]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    out = out.decode('utf-8')

    return isl.Schedule(out)

def add(v1, v2, c):
    """
    v1 + c * v2
    """
    return [v1_ele + v2_ele * c for v1_ele, v2_ele in zip(v1, v2)]


# def multiplyWithCommonDenominator(v):
#     """
#     Multiplies each component of the given vector of rational numbers with
#     their common denominator. The result is a vector of integers.
#     """
#
#     def find_gcd(v):
#         x = reduce(gcd, v)
#         return x
#
#     d = find_gcd(v)
#     if d == 0:
#         d = 1
#     return [int(x / d) for x in v]


def multiplyWithCommonDenominator(numbers):
    """
    Multiplies each component of the given vector of rational numbers with
    their common denominator. The result is a vector of integers.
    """
    def lcm(a, b):
        return a * b // gcd(a, b)

    if all(n == 0 for n in numbers):
        return [int(n) for n in numbers]

    numbers = [str(n) for n in numbers]
    fractions = [Fraction(n).limit_denominator() for n in numbers]
    multiple  = reduce(lcm, [f.denominator for f in fractions])
    ints      = [f * multiple for f in fractions]
    divisor   = reduce(gcd, ints)
    divisor = divisor * (-1) if divisor < 0 else divisor

    return [int(n / divisor) for n in ints]

def extract_jscop(scop_file):
    with open(scop_file) as file:
        jsonp = json.load(file)
        scop = createScopFromDeserializedJson(jsonp)
    return jsonp, scop