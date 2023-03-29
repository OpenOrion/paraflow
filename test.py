#
# script to design and generate mesh for conical nozzle and its corresponding 
# rao bell equivalent contours in .su2 format for CFD analysis.   
#
# Inputs
#     area_ratio, throat_radius, length_percentage
#     (length_percentage used for bell nozzle calculations)
# Outputs
#     conical nozzle : conical_nozzle_cgrid.su2
#     bell nozzle    : bell_nozzle_cgrid.su2
# 
# Author
#     Ravi
#

import gmsh
import math
import warnings
import numpy as np
from bisect import bisect_left

#  area_ratio, throat_radius
def mesh_conical_nozzle(aratio, Rt):
	# half cone angle
	half_cone_angle = 15
	alpha           = math.radians(half_cone_angle)
	# exit radius
	Re = Rt * math.sqrt(aratio)	
	# entrant functions
	entrant_angle  	= -135 # -135 -> yi = 601.64
	ea_radian 		= math.radians(entrant_angle)	
	R1 = 2.0 * Rt
	ea_start 		= ea_radian
	ea_end 			= -math.pi/2	
	x1 = ( R1 * math.cos(ea_start) )
	y1 = ( R1 * math.sin(ea_start) + 3.0 * Rt )
	x2 = ( R1 * math.cos(ea_end) )
	y2 = ( R1 * math.sin(ea_end) + 3.0 * Rt )	
	# exit section
	R1 = 1.5 * Rt	
	L1 = R1 * math.sin(alpha)
	Rn = Rt + R1 * ( 1 - math.cos(alpha) )
	Ln = (Re - Rn) / math.tan(alpha)

	# gmsh packages
	model = gmsh.model
	geo = model.geo	
	mesh = model.mesh
	field = mesh.field	
	# init gmsh
	gmsh.initialize()
	model.add("conical_nozzle")
	# mesh size
	h = 100
	# z co-ordinate
	z = 0
	# inlet origin	
	geo.addPoint( x1, 0, z, h * 0.01, 301 )
	# entry arc points	
	geo.addPoint( x1, y1, z, h * 0.01, 1 )
	geo.addPoint( x2, y2, z, h * 0.01, 2 )
	# inlet arc origin	
	geo.addPoint( 0, 3.0 * Rt, z, h * 0.01, 302 )
	# inlet arc
	geo.addCircleArc( 1, 302, 2, 61 )
	# throat origin point	
	geo.addPoint( 0, 2.5 * Rt, z, h * 0.01, 303 )
	# throat curve (connect inlet last point with bell first point)
	geo.addPoint( L1, Rn, z, h * 0.01, 3 )
	geo.addCircleArc( 2, 303, 3, 62 )
	# symmetry point underneath throat origin
	geo.addPoint( 0, 0, z, h * 0.01, 304 )
	geo.addPoint( L1, 0, z, h * 0.01, 305 )
	# end of cone
	geo.addPoint( L1+Ln, Re, z, h * 0.01, 4 )
	geo.addLine( 3, 4, 40 )
	# symmetry point underneath throat origin
	geo.addPoint( L1+Ln, 0, z, h * 0.01, 306 )	
	# loop lines
	geo.addLine( 301, 1, 41 )
	geo.addLine( 2, 304, 42 )
	geo.addLine( 304, 301, 43 )
	geo.addLine( 3, 305, 44 )
	geo.addLine( 305, 304, 45 )
	# outflow line
	geo.addLine( 4, 306, 46 )	
	geo.addLine( 306, 305, 47 )
	# Curveloop and Surface
	geo.addCurveLoop( [-41, -61, -42, -43], 601 )
	geo.addPlaneSurface( [601], 801 )
	geo.addCurveLoop( [42, -62, -44, -45], 602 )
	geo.addPlaneSurface( [602], 802 )	
	geo.addCurveLoop( [44, -40, -46, -47], 603 )
	geo.addPlaneSurface( [603], 803 )	
	# synchronize
	geo.synchronize()

	# transfinite
	numCellsX = 50 
	gradingX = 1.09; gradingY = 1.0;

	numCellsY = 10
	mesh.setTransfiniteCurve(41, numCellsX, meshType="Progression", coef=-gradingX)
	mesh.setTransfiniteCurve(42, numCellsX, meshType="Progression", coef=1/gradingX)
	mesh.setTransfiniteCurve(61, numCellsY, meshType="Progression", coef=gradingY)
	mesh.setTransfiniteCurve(43, numCellsY, meshType="Progression", coef=gradingY)		
	mesh.setTransfiniteSurface( 801, cornerTags=[301, 304, 2, 1] ) 
	numCellsY = 5
	mesh.setTransfiniteCurve(42, numCellsX, meshType="Progression", coef=gradingX)
	mesh.setTransfiniteCurve(44, numCellsX, meshType="Progression", coef=1/gradingX)
	mesh.setTransfiniteCurve(62, numCellsY, meshType="Progression", coef=gradingY)
	mesh.setTransfiniteCurve(45, numCellsY, meshType="Progression", coef=gradingY)		
	mesh.setTransfiniteSurface( 802, cornerTags=[304, 305, 3, 2] ) 	
	numCellsY = 50 
	mesh.setTransfiniteCurve(44, numCellsX, meshType="Progression", coef=gradingX)
	mesh.setTransfiniteCurve(46, numCellsX, meshType="Progression", coef=1/-gradingX)
	mesh.setTransfiniteCurve(40, numCellsY, meshType="Progression", coef=gradingY)
	mesh.setTransfiniteCurve(47, numCellsY, meshType="Progression", coef=gradingY)		
	mesh.setTransfiniteSurface( 803, cornerTags=[305, 306, 4, 3] ) 	

	# boundaries
	dim = 1
	gmsh.model.addPhysicalGroup(dim, [61, 62, 40], 1001)
	gmsh.model.setPhysicalName(dim, 1001, "WALL")
	gmsh.model.addPhysicalGroup(dim, [41], 1002)
	gmsh.model.setPhysicalName(dim, 1002, "INFLOW")
	gmsh.model.addPhysicalGroup(dim, [46], 1003)
	gmsh.model.setPhysicalName(dim, 1003, "OUTFLOW")
	gmsh.model.addPhysicalGroup(dim, [43, 45, 47], 1004)
	gmsh.model.setPhysicalName(dim, 1004, "SYMMETRY")
	dim = 2
	gmsh.model.addPhysicalGroup(dim, [801, 802, 803], 2001)
	gmsh.model.setPhysicalName(dim, 2001, "Plane surface")
	
	# generate mesh
	gmsh.option.setNumber("Mesh.RecombineAll", 1)
	gmsh.option.setNumber("General.Terminal", 1)
	gmsh.option.setNumber("Mesh.Smoothing", 100)
	gmsh.option.setNumber("Mesh.Algorithm", 5) # delquad
	mesh.generate(2)
	mesh.refine()
	gmsh.write('conical_nozzle_cgrid.geo_unrolled')
	gmsh.finalize()	
	# return		
	return

# exit mach number, area_ratio, throat_radius, length percentage, 
def mesh_bell_nozzle(aratio, Rt, l_percent):
	# upto the nozzle designer, usually -135
	entrant_angle  	= -135
	ea_radian 		= math.radians(entrant_angle)

	# nozzle length percntage
	if l_percent == 60:		Lnp = 0.6
	elif l_percent == 80:	Lnp = 0.8
	elif l_percent == 90:	Lnp = 0.9	
	else:					Lnp = 0.8
	# find wall angles (theta_n, theta_e) for given aratio (ar)		
	angles = find_wall_angles(aratio, throat_radius, l_percent)
	# wall angles
	nozzle_length = angles[0]; theta_n = angles[1]; theta_e = angles[2];

	data_intervel  	= 200
	# entrant functions
	ea_start 		= ea_radian
	ea_end 			= -math.pi/2	
	x1 = ( 1.5 * Rt * math.cos(ea_start) )
	y1 = ( 1.5 * Rt * math.sin(ea_start) + 2.5 * Rt )
	x2 = ( 1.5 * Rt * math.cos(ea_end) )
	y2 = ( 1.5 * Rt * math.sin(ea_end) + 2.5 * Rt )	

	# bell section
	# Nx, Ny-N is defined by [Eqn. 5] setting the angle to (θn – 90)
	Nx = 0.382 * Rt * math.cos(theta_n - math.pi/2)
	Ny = 0.382 * Rt * math.sin(theta_n - math.pi/2) + 1.382 * Rt 
	# Ex - [Eqn. 3], and coordinate Ey - [Eqn. 2]
	Ex = Lnp * ( (math.sqrt(aratio) - 1) * Rt )/ math.tan(math.radians(15) )
	Ey = math.sqrt(aratio) * Rt 
	# gradient m1,m2 - [Eqn. 8]
	m1 = math.tan(theta_n);  m2 = math.tan(theta_e);
	# intercept - [Eqn. 9]
	C1 = Ny - m1*Nx;  C2 = Ey - m2*Ex;
	# intersection of these two lines (at point Q)-[Eqn.10]
	Qx = (C2 - C1)/(m1 - m2)
	Qy = (m1*C2 - m2*C1)/(m1 - m2)	
	
	# Selecting equally spaced divisions between 0 and 1 produces 
	# the points described earlier in the graphical method
	# The bell is a quadratic Bézier curve, which has equations:
	# x(t) = (1 − t)^2 * Nx + 2(1 − t)t * Qx + t^2 * Ex, 0≤t≤1
	# y(t) = (1 − t)^2 * Ny + 2(1 − t)t * Qy + t^2 * Ey, 0≤t≤1 [Eqn. 6]		
	int_list = np.linspace(0, 1, data_intervel)
	xbell = []; ybell = [];
	for t in int_list:		
		xbell.append( ((1-t)**2)*Nx + 2*(1-t)*t*Qx + (t**2)*Ex )
		ybell.append( ((1-t)**2)*Ny + 2*(1-t)*t*Qy + (t**2)*Ey )

	# gmsh packages
	model = gmsh.model
	geo = model.geo	
	mesh = model.mesh
	field = mesh.field	
	# init gmsh
	gmsh.initialize()
	model.add("bell_nozzle")
	# mesh size
	h = 100
	# z co-ordinate
	z = 0
	# inlet	
	geo.addPoint( x1, 0, z, h * 0.01, 301 )	
	geo.addPoint( x1, y1, z, h * 0.01, 1 )
	geo.addPoint( x2, y2, z, h * 0.01, 2 )
	# inlet arc	
	geo.addPoint( 0, 2.5 * Rt, z, h * 0.01, 302 )
	geo.addCircleArc( 1, 302, 2, 61 )

	# bell portion points
	point_id = 3;  points = [];
	h_spline = h*0.01/len(xbell)
	for i, (x, y) in enumerate(zip(xbell, ybell)):
		geo.addPoint(x, y, z, h_spline, point_id + i)
		points.append(point_id + i)

	# throat origin point	
	geo.addPoint( 0, 1.382 * Rt, z, h * 0.01, 303 )
	geo.addCircleArc( 2, 303, points[0], 62 )
	# bell curve	
	geo.addSpline( points, 63 )
	# end point	
	end_bell_point = points[-1] # { xbell[-1], ybell[-1] }
	end_symm_point = end_bell_point + 1
	geo.addPoint( xbell[-1], 0, z, h * 0.01, end_symm_point )	
	geo.addLine( 301, 1, 41 )
	geo.addLine( end_bell_point, end_symm_point, 42 )
	# symmetry lines
	geo.addPoint( x2, 0, z, h * 0.01, 304 )	
	geo.addPoint( xbell[0], 0, z, h * 0.01, 305 ) 
	geo.addLine( end_symm_point, 305, 43 )
	geo.addLine( 305, 304, 44 )
	geo.addLine( 304, 301, 45 )
	geo.addLine( 2, 304, 46 )
	geo.addLine( 305, points[0], 47 )		
	# Curveloop and Surface
	geo.addCurveLoop( [-41, -61, -46, -45], 601 )
	geo.addPlaneSurface( [601], 801 )
	geo.addCurveLoop( [-46, 62, -47, 44], 602 )
	geo.addPlaneSurface( [602], 802 )	
	geo.addCurveLoop( [47, 63, 42, 43], 603 )
	geo.addPlaneSurface( [603], 803 )
	# synchronize
	geo.synchronize()

	# transfinite
	numCellsX = 30 #50 
	gradingX = 1.09; gradingY = 1.0;

	numCellsY = 7 #10
	mesh.setTransfiniteCurve(41, numCellsX, meshType="Progression", coef=-gradingX)
	mesh.setTransfiniteCurve(46, numCellsX, meshType="Progression", coef=1/gradingX)
	mesh.setTransfiniteCurve(61, numCellsY, meshType="Progression", coef=gradingY)
	mesh.setTransfiniteCurve(45, numCellsY, meshType="Progression", coef=gradingY)		
	mesh.setTransfiniteSurface( 801, cornerTags=[301, 304, 2, 1] ) 
	numCellsY = 3 #5
	mesh.setTransfiniteCurve(46, numCellsX, meshType="Progression", coef=gradingX)
	mesh.setTransfiniteCurve(47, numCellsX, meshType="Progression", coef=1/gradingX)
	mesh.setTransfiniteCurve(62, numCellsY, meshType="Progression", coef=gradingY)
	mesh.setTransfiniteCurve(44, numCellsY, meshType="Progression", coef=gradingY)		
	mesh.setTransfiniteSurface( 802, cornerTags=[304, 305, 3, 2] ) 	
	numCellsY = 30 #50
	mesh.setTransfiniteCurve(47, numCellsX, meshType="Progression", coef=1/gradingX)
	mesh.setTransfiniteCurve(42, numCellsX, meshType="Progression", coef=gradingX)
	mesh.setTransfiniteCurve(63, numCellsY, meshType="Progression", coef=gradingY)
	mesh.setTransfiniteCurve(43, numCellsY, meshType="Progression", coef=gradingY)		
	mesh.setTransfiniteSurface( 803, cornerTags=[305, end_symm_point, end_bell_point, 3] ) 	
	
	# boundaries
	dim = 1
	gmsh.model.addPhysicalGroup(dim, [61, 62, 63], 1001)
	gmsh.model.setPhysicalName(dim, 1001, "WALL")
	gmsh.model.addPhysicalGroup(dim, [41], 1002)
	gmsh.model.setPhysicalName(dim, 1002, "INFLOW")
	gmsh.model.addPhysicalGroup(dim, [42], 1003)
	gmsh.model.setPhysicalName(dim, 1003, "OUTFLOW")
	gmsh.model.addPhysicalGroup(dim, [45, 44, 43], 1004)
	gmsh.model.setPhysicalName(dim, 1004, "SYMMETRY")
	dim = 2
	gmsh.model.addPhysicalGroup(dim, [801, 802, 803], 2001)
	gmsh.model.setPhysicalName(dim, 2001, "Plane surface")
		
	# generate mesh
	gmsh.option.setNumber("Mesh.RecombineAll", 1)
	gmsh.option.setNumber("General.Terminal", 1)
	gmsh.option.setNumber("Mesh.Smoothing", 100)
	gmsh.option.setNumber("Mesh.Algorithm", 5) # delquad	
	mesh.generate(2)
	mesh.refine()
	gmsh.write('bell_nozzle_cgrid.geo_unrolled')
	gmsh.finalize()	
	# return
	return


# Liquid Rocket Engine Nozzles NASA SP-8120
# https://ntrs.nasa.gov/search.jsp?R=19770009165
# find wall angles (theta_n, theta_e) in radians for given aratio (ar)
def find_wall_angles(ar, Rt, l_percent = 80 ):
	# wall-angle empirical data
	aratio 		= [ 4,    5,    10,   20,   30,   40,   50,   100]
	theta_n_60 	= [20.5, 20.5, 16.0, 14.5, 14.0, 13.5, 13.0, 11.2]
	theta_n_80 	= [21.5, 23.0, 26.3, 28.8, 30.0, 31.0, 31.5, 33.5]
	theta_n_90 	= [20.0, 21.0, 24.0, 27.0, 28.5, 29.5, 30.2, 32.0]
	theta_e_60 	= [26.5, 28.0, 32.0, 35.0, 36.2, 37.1, 35.0, 40.0]
	theta_e_80 	= [14.0, 13.0, 11.0,  9.0,  8.5,  8.0,  7.5,  7.0]
	theta_e_90 	= [11.5, 10.5,  8.0,  7.0,  6.5,  6.0,  6.0,  6.0]	

	# nozzle length
	f1 = ( (math.sqrt(ar) - 1) * Rt )/ math.tan(math.radians(15) )
	
	if l_percent == 60:
		theta_n = theta_n_60; theta_e = theta_e_60;
		Ln = 0.8 * f1
	elif l_percent == 80:
		theta_n = theta_n_80; theta_e = theta_e_80;
		Ln = 0.8 * f1		
	elif l_percent == 90:
		theta_n = theta_n_90; theta_e = theta_e_90;	
		Ln = 0.9 * f1	
	else:
		theta_n = theta_n_80; theta_e = theta_e_80;		
		Ln = 0.8 * f1

	# find the nearest ar index in the aratio list
	x_index, x_val = find_nearest(aratio, ar)
	# if the value at the index is close to input, return it
	if round(aratio[x_index], 1) == round(ar, 1):
		return theta_n[x_index], theta_e[x_index]

	# check where the index lies, and slice accordingly
	if (x_index>2):
		# slice couple of middle values for interpolation
		ar_slice = aratio[x_index-2:x_index+2]		
		tn_slice = theta_n[x_index-2:x_index+2]
		te_slice = theta_e[x_index-2:x_index+2]
		# find the tn_val for given ar
		tn_val = interpolate(ar_slice, tn_slice, ar)	
		te_val = interpolate(ar_slice, te_slice, ar)	
	elif( (len(aratio)-x_index) <= 1):
		# slice couple of values initial for interpolation
		ar_slice = aratio[x_index-2:len(x_index)]		
		tn_slice = theta_n[x_index-2:len(x_index)]
		te_slice = theta_e[x_index-2:len(x_index)]
		# find the tn_val for given ar
		tn_val = interpolate(ar_slice, tn_slice, ar)	
		te_val = interpolate(ar_slice, te_slice, ar)	
	else:
		# slice couple of end values for interpolation
		ar_slice = aratio[0:x_index+2]		
		tn_slice = theta_n[0:x_index+2]
		te_slice = theta_e[0:x_index+2]
		# find the tn_val for given ar
		tn_val = interpolate(ar_slice, tn_slice, ar)	
		te_val = interpolate(ar_slice, te_slice, ar)						

	return Ln, math.radians(tn_val), math.radians(te_val)

# simple linear interpolation
def interpolate(x_list, y_list, x):
	if any(y - x <= 0 for x, y in zip(x_list, x_list[1:])):
		raise ValueError("x_list must be in strictly ascending order!")
	intervals = zip(x_list, x_list[1:], y_list, y_list[1:])
	slopes = [(y2 - y1) / (x2 - x1) for x1, x2, y1, y2 in intervals]

	if x <= x_list[0]:
		return y_list[0]
	elif x >= x_list[-1]:
		return y_list[-1]
	else:
		i = bisect_left(x_list, x) - 1
		return y_list[i] + slopes[i] * (x - x_list[i])

# find the nearest index in the list for the given value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]  

# design and generate mesh
def generate_nozzle_mesh(aratio, throat_radius, l_percent=80):
	# design and mesh conical_nozzle_contour
	mesh_conical_nozzle(aratio, throat_radius)

	# design and mesh rao_bell_nozzle_contour
	# mesh_bell_nozzle(aratio, throat_radius, l_percent)
	# return

# __main method__
if __name__=="__main__":

	# PSLV - stage PS1 - S139 
	throat_radius = 836/2.0 
	exit_radius = 2377/2.0
	aratio = math.pow(exit_radius,2) / math.pow(throat_radius,2) # Ae / At
	
	# PSLV - stage PS3
	# throat_radius = 100.52 
	# exit_radius = 718.0
	# aratio = math.pow(exit_radius,2) / math.pow(throat_radius,2)  # Ae / At
	
	# nozzle length percentage (for bell nozzle) from 15 degree conical nozzle (available 60, 80, 90 values)
	l_percent = 80
	
	# design and generate mesh for conical nozzle and its corresponding rao bell equivalent
	generate_nozzle_mesh(aratio, throat_radius, l_percent)


