import ROOT

import math
import numpy as np
from scipy.spatial import cKDTree
import statistics
import sys


def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian


def calculate_bow_and_thickness(name, data):
    """
    Performs the calculations necessary to measure the bow and thickness of a carbon fiber plate

    :param name: The name of the side being analyzed. Typically "top" or "bottom"
    :param data: The data being analyzed for the corresponding side
    :return: The fitted TF2 plane to the data to be used for analysis
    """

    #
    # Perform data manipulations to make analysis much easier.
    #
    theta = math.atan(35 / 270)  # can rotate by theta or theta + 90 degrees
    rotation_matrix = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    rotated_data = np.transpose(
        np.matmul(rotation_matrix, np.transpose(data[:, [0, 2]])))  # rotate by -theta
    data[:, 0] = rotated_data[:, 0]
    data[:, 2] = rotated_data[:, 1]

    data[:, 0] = data[:, 0] - np.amin(data[:, 0])  # zero out the x axis
    data[:, 2] = data[:, 2] - np.amin(data[:, 2])  # zero out the y axis

    # Create 3D plane for top or bottom
    plane = ROOT.TF2(name + "_plane", "[0]*x + [1]*y + [2]", np.amin(data[:, 0]), np.amax(data[:, 0]),
                     np.amin(data[:, 2]), np.amax(data[:, 2]))
    plane.SetParameters(1, 1, -30)

    # Do the plotting and fitting...
    g_birds_eye = ROOT.TGraph2D()
    g_birds_eye.SetTitle("Bird's eye view of " + name + " ; x (mm); y (mm); z (mm)")

    g_xz = ROOT.TGraph()
    g_yz = ROOT.TGraph()
    g_xz.SetTitle(name + "; x (mm); z (mm)")
    g_yz.SetTitle(name + "; y (mm); z (mm)")

    for j in range(len(data)):
        g_birds_eye.SetPoint(j, data[j][0], data[j][2], data[j][1])
        g_xz.SetPoint(j, data[j][0], data[j][1])
        g_yz.SetPoint(j, data[j][2], data[j][1])

    c_birds_eye = ROOT.TCanvas("c_birds_eye", "c_birds_eye", 700, 700)
    c_birds_eye.SetRightMargin(0.15)
    g_birds_eye.Fit(plane)
    g_birds_eye.Draw("colz cont")
    c_birds_eye.SaveAs("c_birds_eye" + name + ".png")
    c_birds_eye.SaveAs("c_birds_eye" + name + ".root")

    c_xz_graph = ROOT.TCanvas("c_xz_graph", "c_xz_graph", 700, 700)
    c_xz_graph.SetLeftMargin(0.15)
    x_plane = ROOT.TF12("x_plane", plane, 150, "x")
    g_xz.Draw("AP same")
    x_plane.Draw("same")
    c_xz_graph.SaveAs("c_" + name + "_xz_graph.png")
    c_xz_graph.SaveAs("c_" + name + "_xz_graph.root")

    c_yz_graph = ROOT.TCanvas("c_yz_graph", "c_yz_graph", 700, 700)
    c_yz_graph.SetLeftMargin(0.15)
    y_plane = ROOT.TF12("y_plane", plane, 150, "y")
    g_yz.Draw("AP same")
    y_plane.Draw("same")
    c_yz_graph.SaveAs("c_" + name + "_yz_graph.png")
    c_yz_graph.SaveAs("c_" + name + "_yz_graph.root")

    return plane, data


ROOT.gStyle.SetOptStat(0000)

filename = sys.argv[1]
bottom_data = np.genfromtxt(filename + "_Bottom.txt", delimiter=';', dtype=float, skip_header=3)
top_data = np.genfromtxt(filename + "_Top.txt", delimiter=';', dtype=float, skip_header=3)

bottom_plane, bottom_data = calculate_bow_and_thickness("bottom", bottom_data)
top_plane, top_data = calculate_bow_and_thickness("top", top_data)

# 3D Plane: f(x,y) = a*x + b*y + c
top_a = top_plane.GetParameter(0)
bottom_a = bottom_plane.GetParameter(0)
top_b = top_plane.GetParameter(1)
bottom_b = bottom_plane.GetParameter(1)
top_c = top_plane.GetParameter(2)
bottom_c = bottom_plane.GetParameter(2)

average_a = (top_a + bottom_a) / 2
average_b = (top_b + bottom_b) / 2
average_c = (top_c + bottom_c) / 2
print("top shape is " + str(top_data.shape))
print("bottom shape is " + str(bottom_data.shape))
mean_plane = ROOT.TF2("mean_plane", "[0]*x + [1]*y + [2]",
                      np.amin(np.array([np.amin(top_data[:, 0]), np.amin(bottom_data[:, 0])])),
                      np.amax(np.array([np.amin(top_data[:, 0]), np.amin(bottom_data[:, 0])])),
                      np.amin(np.array([np.amin(top_data[:, 2]), np.amin(bottom_data[:, 2])])),
                      np.amax(np.array([np.amin(top_data[:, 2]), np.amin(bottom_data[:, 2])])))
mean_plane.SetParameter(0, average_a)
mean_plane.SetParameter(1, average_b)
mean_plane.SetParameter(2, average_c)
print("The equation for the mean plane is " + str(format(average_a, '0.7f')) + "*x + " + str(format(average_b, '0.5f'))
      + "*y + " + str(format(average_c, '0.3f')))

top_z_dist_from_mean_plane = []
bottom_z_dist_from_mean_plane = []

for i in range(len(top_data)):
    top_z_dist_from_mean_plane.append(top_data[i][1] - mean_plane.Eval(top_data[i][0], top_data[i][2]))
for i in range(len(bottom_data)):
    bottom_z_dist_from_mean_plane.append(bottom_data[i][1] - mean_plane.Eval(bottom_data[i][0], bottom_data[i][2]))

top_z_dist_from_mean_plane = np.array(top_z_dist_from_mean_plane)
bottom_z_dist_from_mean_plane = np.array(bottom_z_dist_from_mean_plane)

idx = cKDTree(bottom_data[:, 0:3:2]).query(top_data[:, 0:3:2], k=1)[1]
thickness = np.reshape(np.concatenate(((top_data[:, 0] + bottom_data[idx, 0]) / 2, (top_data[:, 2] + bottom_data[idx, 2]) / 2, top_z_dist_from_mean_plane[:] + bottom_z_dist_from_mean_plane[idx]), axis=0), (3, len(top_data))).T

mean_total_thickness = statistics.mean(thickness[:, 2])
stdev_total_thickness = statistics.stdev(thickness[:, 2])

g_Thickness = ROOT.TGraph2D()
g_Thickness.SetTitle("; x (cm); y (cm); z (mm)")
for i in range(len(thickness)):
    g_Thickness.SetPoint(i, thickness[i][0], thickness[i][1], thickness[i][2])

label_with_edge = ROOT.TPaveLabel()
c_Thickness_graph = ROOT.TCanvas("c_Thickness_graph", "c_Thickness_graph", 700, 700)
g_Thickness.Draw("surf1")
c_Thickness_graph.SaveAs("c_Thickness_graph.png")
c_Thickness_graph.SaveAs("c_Thickness_graph.root")

label_with_edge = ROOT.TPaveLabel(0.1, 0.92, 0.9, 0.96, "With \hspace Edge:\Delta z = " + str(format(mean_total_thickness, '0.4f')) + " \pm " + str(format(stdev_total_thickness, '0.4f')) + "\hspace mm")
label_with_edge.SetFillColor(16)
label_with_edge.SetTextFont(52)

c_Thickness_2D = ROOT.TCanvas("c_Thickness_2D", "c_Thickness_2D", 700, 700)
c_Thickness_2D.SetRightMargin(0.15)
g_Thickness.Draw("colz cont")
label_with_edge.Paint("NDC")
label_with_edge.Draw()
c_Thickness_2D.SaveAs("c_Thickness_2D.png")

input()