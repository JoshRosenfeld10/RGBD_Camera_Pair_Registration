import logging
import os
from typing import Annotated, Optional
from urllib.request import urlretrieve

import numpy as np
import slicer, qt, vtk, ctk
import math
import numpy
import cv2
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

import FiducialsToModelRegistration


try:
    import torch
except ModuleNotFoundError:
    slicer.util.pip_install("torch torchvision --index-url https://download.pytorch.org/whl/cu126")

try:
    from segment_anything import sam_model_registry, SamPredictor
except ModuleNotFoundError:
    slicer.util.pip_install("segment_anything")
    from segment_anything import sam_model_registry, SamPredictor


#
# RGBD_Camera_Pair_Registration
#


class RGBD_Camera_Pair_Registration(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("RGBD Camera Pair Registration")
        self.parent.categories = ["RGBD Tracking"]
        self.parent.dependencies = []
        self.parent.contributors = ["Josh Rosenfeld (Perk Lab, Queen's University)"]  # TODO: replace with "Firstname Lastname (Organization)"
        self.parent.helpText = _("""
This module registers bounding boxes paired RGB-D cameras into the same coordinate frame.
""")


#
# RGBD_Camera_Pair_RegistrationParameterNode
#


@parameterNodeWrapper
class RGBD_Camera_Pair_RegistrationParameterNode:
    pass

#
# RGBD_Camera_Pair_RegistrationWidget
#


class RGBD_Camera_Pair_RegistrationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self._updatingGUIFromParameterNode = False

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/RGBD_Camera_Pair_Registration.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        self.ui.videoIDComboBox.addItem("Select video ID")

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.sequenceBrowser.setMRMLScene(slicer.mrmlScene)
        self.ui.rightCameraTransform.setMRMLScene(slicer.mrmlScene)
        self.ui.aboveCameraTransform.setMRMLScene(slicer.mrmlScene)
        self.ui.rightBoundingBoxSequence.setMRMLScene(slicer.mrmlScene)
        self.ui.aboveBoundingBoxSequence.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = RGBD_Camera_Pair_RegistrationLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.DirectoryButton.connect('directorySelected(QString)', self.onDatasetSelected)
        self.ui.depthToRASPushButton.connect('clicked(bool)', self.onGetDepthToRAS)
        self.ui.registerBoundingBoxButton.connect('clicked(bool)', self.onRegisterBoundingBoxButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def onDatasetSelected(self):
        for i in range(self.ui.videoIDComboBox.count, 0, -1):
            self.ui.videoIDComboBox.removeItem(i)
        self.currentDatasetName = os.path.basename(self.ui.DirectoryButton.directory)
        self.videoPath = self.ui.DirectoryButton.directory
        self.addVideoIDsToComboBox()

    def addVideoIDsToComboBox(self):
        for i in range(1, self.ui.videoIDComboBox.count + 1):
            self.ui.videoIDComboBox.removeItem(i)
        videoIDList = os.listdir(self.videoPath)
        self.videoIDList = [dir for dir in videoIDList if dir.rfind(".") == -1]  # get only directories
        self.ui.videoIDComboBox.addItems(self.videoIDList)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode: Optional[RGBD_Camera_Pair_RegistrationParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)

    def onGetDepthToRAS(self):
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            self.logic.getDepthToRAS(
                self.ui.rightOrAboveComboBox.currentText
            )

    def onRegisterBoundingBoxButton(self):
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            self.logic.registerBoundingBox(
                self.ui.sequenceBrowser.currentNode(),
                self.ui.rightCameraTransform.currentNode(),
                self.ui.aboveCameraTransform.currentNode(),
                self.ui.rightBoundingBoxSequence.currentNode(),
                self.ui.aboveBoundingBoxSequence.currentNode()
            )



#
# RGBD_Camera_Pair_RegistrationLogic
#


class RGBD_Camera_Pair_RegistrationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        self.fid2ModLogic = slicer.util.getModuleLogic("FiducialsToModelRegistration")
        self.phantomModel = None
        self.phantomModelBounds = None
        self.phantomBBox = None
        self.depthToRAS = None
        self.initialDepthToRAS = None
        self.depthNode = None
        self.imgShape = None
        self.depthImage = None
        self.fiducialNode = None
        self.referenceFiducialNode = None
        self.cameraView = "RIGHT"
        self.loadModel()

        # TODO: change so both above and right transforms are computed at once
        # TODO: Move the reference fiducials under the DepthToRAS transform to see a better result

    def loadModel(self):
        sam_checkpoint = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Resources", "sam_vit_h_4b8939.pth")
        if not os.path.exists(sam_checkpoint):
            download_url = ("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
            print("Downloading copy of weights, this may take a few minutes")
            urlretrieve(download_url, sam_checkpoint)
        model_type = "vit_h"
        device = "cuda"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)

    def predict(self, image, bbox):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # check shape to see if image or points list
        # will only have 2 shape dimensions for poitns list, 3 dims for colour image
        # for image, set image to self.image
        # if new image, update attribute
        # if you get points, update points and do predictions again
        original_image_shape = (image.shape[0], image.shape[1])
        self.predictor.set_image(image)
        bbox_prompt = numpy.array(
            [int(bbox["xmin"]), int(bbox["ymin"]), int(bbox["xmax"]), int(bbox["ymax"])])
        masks, scores, logits = self.predictor.predict(
            box=bbox_prompt,
            multimask_output=False
        )
        bestMask = masks[numpy.argmax(scores), :, :]
        bestMask = numpy.where(bestMask == True, 1.0, 0.0)

        # bestMask = bestMask[int(bbox["ymin"]):int(bbox["ymax"]),int(bbox["xmin"]):int(bbox["xmax"])]
        try:
            mask = slicer.util.getNode("Mask")
            slicer.util.updateVolumeFromArray(mask, bestMask)
        except:
            mask = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
            mask.SetName("Mask")
            slicer.util.updateVolumeFromArray(mask, bestMask)
        return bestMask


    def getParameterNode(self):
        return RGBD_Camera_Pair_RegistrationParameterNode(super().getParameterNode())

    def registerBoundingBox(self, sequenceBrowser, rightCameraTransform, aboveCameraTransform, rightBoundingBoxSequence, aboveBoundingBoxSequence):
        roiSequence = self.getOrCreateROINodes("TEST")

        # Link ROI to the browser
        if sequenceBrowser.GetSequenceNode(roiSequence) is None:
            sequenceBrowser.AddSynchronizedSequenceNode(roiSequence)
            # sequenceBrowser.AddProxyNode(roiProxy, roiSequence)

        # Proxies for the two bounding box sequences
        proxyRight = sequenceBrowser.GetProxyNode(rightBoundingBoxSequence)
        proxyAbove = sequenceBrowser.GetProxyNode(aboveBoundingBoxSequence)
        if proxyRight is None or proxyAbove is None:
            raise RuntimeError("Sequence Browser must reference both bounding box sequences.")

        # Get number of frames to process
        nRight = rightBoundingBoxSequence.GetNumberOfDataNodes()
        nAbove = aboveBoundingBoxSequence.GetNumberOfDataNodes()
        if nRight <= nAbove:
            masterSequence = rightBoundingBoxSequence
            n = nRight
        else:
            masterSequence = aboveBoundingBoxSequence
            n = nAbove

        roiSequence.RemoveAllDataNodes()

        # Remember parameters of last good 3D box to use when one view is occluded
        lastPmin = None
        lastPmax = None
        lastRLength = None  # length along R
        lastSLength = None  # length along S

        # Process all frames
        for i in range(n):
            sequenceBrowser.SetSelectedItemNumber(i)
            indexValue = masterSequence.GetNthIndexValue(i)  # the time/frame index label of current frame

            # Get 4 corners of bounding box from each view in RAS
            pointsRight = self.getBoundingBoxInRAS(proxyRight, rightCameraTransform, minSize=1.0)
            pointsAbove = self.getBoundingBoxInRAS(proxyAbove, aboveCameraTransform, minSize=1.0)

            # Check if bounding box is valid for current frame
            rightOK = pointsRight.shape[0] >= 4
            aboveOK = pointsAbove.shape[0] >= 4

            if rightOK and aboveOK:
                # Both cameras working
                pmin, pmax = self.tightAxisAlignedBoundingBox([pointsRight, pointsAbove])
                lastPmin, lastPmax = pmin, pmax
                lastRLength = pmax[0] - pmin[0]
                lastSLength = pmax[2] - pmin[2]

            elif rightOK and lastPmin is not None and lastPmax is not None and lastRLength is not None and lastSLength is not None:
                # ABOVE is occluded; build ROI from right and last known position of ROI
                print(f"ABOVE OCCLUDED; USING RIGHT for frame {i}")
                rmin, rmax = self.tightAxisAlignedBoundingBox([pointsRight])

                cR = 0.5 * (rmin + rmax)  # center from right
                cR[0] -= 0.5 * lastRLength
                pmin = np.array([
                    cR[0] - 0.5  * lastRLength,
                    rmin[1],
                    rmin[2]
                ], dtype=float)
                pmax = np.array([
                    cR[0] + 0.5 * lastRLength,
                    rmax[1],
                    rmax[2]
                ], dtype=float)

                lastPmin, lastPmax = pmin, pmax
                lastSLength = pmax[2] - pmin[2]

            elif aboveOK and lastPmin is not None and lastPmax is not None:
                # RIGHT is occluded; build ROI from above and last known position of ROI
                print(f"RIGHT OCCLUDED; USING ABOVE for frame {i}")
                amin, amax = self.tightAxisAlignedBoundingBox([pointsAbove])

                cA = 0.5 * (amin + amax)  # center from anterior
                cA[2] -= 0.5 * lastSLength
                pmin = np.array([
                    amin[0],
                    amin[1],
                    cA[2] - 0.5 * lastSLength
                ], dtype=float)
                pmax = np.array([
                    amax[0],
                    amax[1],
                    cA[2] + 0.5 * lastSLength
                ], dtype=float)

                lastPmin, lastPmax = pmin, pmax
                lastRLength = pmax[0] - pmin[0]

            elif lastPmin is not None and lastPmax is not None:
                # Neither camera view working, reuse last
                pmin, pmax = lastPmin, lastPmax

            else:
                continue

            center = 0.5 * (pmin + pmax)
            size = pmax - pmin

            # Make a data node to store in the sequence
            dataNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "TempROINode")
            dataNode.SetCenter(center.tolist())
            dataNode.SetSize(size.tolist())

            # Store temporary data node into ROI sequence
            roiSequence.SetDataNodeAtValue(dataNode, indexValue)

            # Remove the temporary data node from the scene
            slicer.mrmlScene.RemoveNode(dataNode)
            # self.updateROIFromBounds(roiProxy, pmin, pmax)

    def updateROIFromBounds(self, roiProxy, pmin, pmax):
        center = 0.5 * (pmin + pmax)
        size = pmax - pmin
        roiProxy.SetCenter(center.tolist())
        roiProxy.SetSize(size.tolist())

    def tightAxisAlignedBoundingBox(self, pointsList):
        pts = np.vstack(pointsList)
        return pts.min(axis=0), pts.max(axis=0)

    def getBoundingBoxInRAS(self, markupsNode, transformNode, minSize=1.0):
        n = markupsNode.GetNumberOfControlPoints()
        if n == 0:
            return np.zeros((0,3))
        gt = vtk.vtkGeneralTransform()
        transformNode.GetTransformToWorld(gt)
        pts = np.zeros((n,3), dtype=float)
        p = [0.0,0.0,0.0]
        for i in range(n):
            markupsNode.GetNthControlPointPosition(i, p)
            pw = gt.TransformPoint(p)
            pts[i,:] = pw

        # Check size of bounding box
        pmin = pts.min(axis=0)
        pmax = pts.max(axis=0)
        spans = pmax - pmin
        if (spans[0] < minSize) and (spans[1] < minSize) and (spans[2] < minSize):
            return np.zeros((0,3))

        return pts

    def getOrCreateROINodes(self, classname):
        # roiProxy = slicer.util.getFirstNodeByClassByName("vtkMRMLMarkupsROINode", f"{classname.upper()}_ROI") \
        #     if slicer.util.getFirstNodeByClassByName(
        #     "vtkMRMLMarkupsROINode", f"{classname.upper()}_ROI" ) \
        #     else slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", f"{classname.upper()}_ROI")
        roiSequence = slicer.util.getFirstNodeByClassByName("vtkMRMLSequenceNode", f"{classname.upper()}_ROI_SEQUENCE") \
            if slicer.util.getFirstNodeByClassByName(
            "vtkMRMLSequenceNode", f"{classname.upper()}_ROI_SEQUENCE") \
            else slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", f"{classname.upper()}_ROI_SEQUENCE")
        return roiSequence

    def convert_pixels_to_mm(self,x_pixels, y_pixels, z_mm, principle_pt=[314.273, 251.183], focal_length=592.667):
        x_mm = ((x_pixels - principle_pt[0])/focal_length)*(z_mm+0.578)
        y_mm = ((y_pixels - principle_pt[1])/focal_length)*(z_mm+0.578)
        return (x_mm, y_mm)

    def getDepthToRAS(self, camera: str):
        self.cameraView = camera
        print(self.cameraView)

        # TODO: change this so the phantom model can actually be selected in the UI
        self.phantomModel = slicer.util.getFirstNodeByName("BluePhantom")
        # self.phantomModel = slicer.util.getFirstNodeByName("SkinModel")

        # Get the RAS bounds of the phantom model in the scene
        # We treat the phantom model in the scene as the RAS coordinates of the phantom
        self.phantomModelBounds = [0, 0, 0, 0, 0, 0]  # (xmin, xmax, ymin, ymax, zmin, zmax)
        self.phantomModel.GetRASBounds(self.phantomModelBounds)

        # Get the bounding box of the phantom on the first frame of the video
        # Note that OpenCV and Slicer don't use the same (0,0) point of an image (openCV (0,0) = top left, slicer (0,0) = bottom left (I THINK; DOUBLE CHECK))
        # self.phantomBBox = {"class": "phantom", "xmin": 0, "xmax": 640, "ymin": 0, "ymax": 480}
        self.phantomBBox = {"class": "phantom", "xmin": 101, "ymin": 55, "xmax": 101 + 534, "ymax": 55 + 349}\
            if self.cameraView == "RIGHT" else {'class': 'phantom', 'xmin': 136, 'ymin': 73, 'xmax': 640, 'ymax': 480}

        # If DepthToRAS transform doesn't exist, create one
        self.depthToRAS = slicer.util.getFirstNodeByName("DepthToRAS")
        if not self.depthToRAS:
            self.depthToRAS = slicer.vtkMRMLLinearTransformNode()
            self.depthToRAS.SetName("DepthToRAS")
            slicer.mrmlScene.AddNode(self.depthToRAS)

        # Get video nodes
        self.depthNode = slicer.util.getFirstNodeByClassByName(
            "vtkMRMLStreamingVolumeNode",
            "Image1DEPTH_Image1DE" if self.cameraView == "RIGHT" else "ImageDEPTH_ImageDEPT"
        )
        self.rgbNode = slicer.util.getFirstNodeByClassByName(
            "vtkMRMLStreamingVolumeNode",
            "Image1RGB_Image1RGB" if self.cameraView == "RIGHT" else "ImageRGB_ImageRGB"
        )

        self.getDepthImage(self.phantomBBox)  # Get the depth image of the first frame of the video (with ROI as phantom bounding box)
        self.getBestComponent(self.phantomModel, self.depthToRAS, self.phantomBBox)  # Try to get depth points of the phantom only
        self.transformFiducialsToModel()
        # self.resizeAndAlignImage()  # Initial registration; align the image corners and phantom bb to the bounds of the phantom model in scene.

        # This is the actual ICP registration after the initial registration
        # self.convertDepthToPoints(self.phantomBBox)
        # self.referenceFiducialNode.SetAndObserveTransformNodeID(self.initialDepthToRAS.GetID())
        # self.referenceFiducialNode.HardenTransform()
        slicer.mrmlScene.Modified()
        # self.updateDepthToRASTransform()
        # self.initialDepthToRAS.SetAndObserveTransformNodeID(self.depthToRAS.GetID())

    def applyTransformToNode(self, node, transformNode):
        node.SetAndObserveTransformNodeID(transformNode.GetID())

    def hardenNodeTransform(self, node):
        slicer.vtkSlicerTransformLogic().hardenTransform(node)
        node.SetAndObserveTransformNodeID(None)

    def makeLinearTransformNodeFromVTKTransform(self, vtkTransformObject, name):
        transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", name)
        m = vtk.vtkMatrix4x4()
        vtkTransformObject.GetMatrix(m)
        transformNode.SetMatrixTransformToParent(m)
        return transformNode

    def transformFiducialsToModel(self):
        fid2ModelLogic = FiducialsToModelRegistration.FiducialsToModelRegistrationLogic()

        # 1. Hard coded pre-alignment
        rotationTransform = vtk.vtkTransform()
        rotationTransform.PostMultiply()
        if self.cameraView == "ABOVE":
            rotationTransform.RotateX(180.0)
            rotationTransform.RotateY(-90.0)
        else:  # RIGHT
            # TODO: this is probably not right, it looks like the right view is inverted
            rotationTransform.RotateX(-90.0)
            rotationTransform.RotateZ(-90.0)

        rotationTransformNode = self.makeLinearTransformNodeFromVTKTransform(rotationTransform, "PreAlignmentRotation")
        self.applyTransformToNode(self.referenceFiducialNode, rotationTransformNode)
        self.hardenNodeTransform(self.referenceFiducialNode)

        # 2. Rigid transform to model
        rigidTransformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "RigidTransform")
        fid2ModelLogic.run(self.referenceFiducialNode, self.phantomModel, rigidTransformNode, transformType=0, numIterations=100)
        self.applyTransformToNode(self.referenceFiducialNode, rigidTransformNode)
        self.hardenNodeTransform(self.referenceFiducialNode)

        # 3. Affine transform to model
        affineTransformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "AffineTransform")
        fid2ModelLogic.run(self.referenceFiducialNode, self.phantomModel, affineTransformNode, transformType=2, numIterations=100)
        self.applyTransformToNode(self.referenceFiducialNode, affineTransformNode)

        # TODO: combine all 3 transforms to create the final DepthToRAS transform


    def updateDepthToRASTransform(self):
        self.fid2ModLogic.run(self.referenceFiducialNode, self.phantomModel, self.depthToRAS, 0, 100)
        # slicer.mrmlScene.Modified()

    def resizeAndAlignImage(self):
        try:
            self.initialDepthToRAS = slicer.util.getNode("InitialDepthToRAS")
        except slicer.util.MRMLNodeNotFoundException:
            self.initialDepthToRAS = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
            self.initialDepthToRAS.SetName("InitialDepthToRAS")
        try:
            modelCornerPoints = slicer.util.getNode("ModelCorners")
            modelCornerPoints.RemoveAllMarkups()
        except slicer.util.MRMLNodeNotFoundException:
            modelCornerPoints = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
            modelCornerPoints.SetName("ModelCorners")

        try:
            imageCornerPoints = slicer.util.getNode("BBoxCorners")
            imageCornerPoints.RemoveAllMarkups()
        except slicer.util.MRMLNodeNotFoundException:
            imageCornerPoints = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
            imageCornerPoints.SetName("BBoxCorners")

        try:
            fullImageCorners = slicer.util.getNode("ImageCorners")
            fullImageCorners.RemoveAllMarkups()
        except slicer.util.MRMLNodeNotFoundException:
            fullImageCorners = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
            fullImageCorners.SetName("ImageCorners")

        # (xmin, xmax, ymin, ymax, zmin, zmax)
        # MODEL FIDUCIALS
        if self.cameraView == "RIGHT":
            modelCornerPoints.AddControlPoint(self.phantomModelBounds[1], self.phantomModelBounds[3],
                                          self.phantomModelBounds[4])
            modelCornerPoints.AddControlPoint(self.phantomModelBounds[1], self.phantomModelBounds[2],
                                          self.phantomModelBounds[5])
            modelCornerPoints.AddControlPoint(self.phantomModelBounds[1], self.phantomModelBounds[3],
                                          self.phantomModelBounds[5])
        else:
            # Above camera view
            modelCornerPoints.AddControlPoint(self.phantomModelBounds[1], self.phantomModelBounds[2],
                                              self.phantomModelBounds[4])
            modelCornerPoints.AddControlPoint(self.phantomModelBounds[0], self.phantomModelBounds[2],
                                              self.phantomModelBounds[5])
            modelCornerPoints.AddControlPoint(self.phantomModelBounds[1], self.phantomModelBounds[2],
                                              self.phantomModelBounds[5])

        # THIS IS UPDATED. RIGHT USES MAX, ABOVE USES MIN
        maxDepth = numpy.max(self.depthImage)
        minDepth = numpy.min(self.depthImage)
        self.abovePlaneDepth = float(minDepth)
        print(f"MIN DEPTH: {minDepth}")

        # PHANTOM BOUNDING BOX FIDUCIALS
        if self.cameraView == "RIGHT":
            imageCornerPoints.AddControlPoint(maxDepth, self.phantomBBox["ymax"], self.phantomBBox["xmin"])
            imageCornerPoints.AddControlPoint(maxDepth, self.phantomBBox["ymin"], self.phantomBBox["xmax"])
            imageCornerPoints.AddControlPoint(maxDepth, self.phantomBBox["ymax"], self.phantomBBox["xmax"])
        else:
            # Above camera view
            imageCornerPoints.AddControlPoint(self.phantomBBox["ymax"], minDepth, self.phantomBBox["xmin"])
            imageCornerPoints.AddControlPoint(self.phantomBBox["ymin"], minDepth, self.phantomBBox["xmax"])
            imageCornerPoints.AddControlPoint(self.phantomBBox["ymax"], minDepth, self.phantomBBox["xmax"])

        # IMAGE FIDUCIALS
        if self.cameraView == "RIGHT":
            fullImageCorners.AddControlPoint(maxDepth, 480, 0)
            fullImageCorners.AddControlPoint(maxDepth, 0, 640)
            fullImageCorners.AddControlPoint(maxDepth, 480, 640)
        else:
            # Above camera view
            fullImageCorners.AddControlPoint(480, minDepth, 0)
            fullImageCorners.AddControlPoint(0, minDepth, 640)
            fullImageCorners.AddControlPoint(480, minDepth, 640)

        fiducialRegistrationNode = slicer.vtkMRMLFiducialRegistrationWizardNode()
        slicer.mrmlScene.AddNode(fiducialRegistrationNode)
        fiducialRegistrationNode.SetAndObserveFromFiducialListNodeId(imageCornerPoints.GetID())
        fiducialRegistrationNode.SetAndObserveToFiducialListNodeId(modelCornerPoints.GetID())
        fiducialRegistrationNode.SetOutputTransformNodeId(self.initialDepthToRAS.GetID())
        fiducialRegistrationNode.SetRegistrationModeToSimilarity()
        fidRegLogic = slicer.util.getModuleLogic("FiducialRegistrationWizard")
        fidRegLogic.UpdateCalibration(fiducialRegistrationNode)
        imageCornerPoints.SetAndObserveTransformNodeID(self.initialDepthToRAS.GetID())
        fullImageCorners.SetAndObserveTransformNodeID(self.initialDepthToRAS.GetID())

        # slicer.mrmlScene.Modified()

    def getVtkImageDataAsOpenCVMat(self, cameraVolume):
        # cameraVolume = self.depthNode

        image = cameraVolume.GetImageData()
        shape = list(cameraVolume.GetImageData().GetDimensions())
        shape.reverse()
        components = image.GetNumberOfScalarComponents()
        if components > 1:
            shape.append(components)
            shape.remove(1)
        imageMat = vtk.util.numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(shape)
        return imageMat

    def convertRGBtoD(self, pixel1):
        is_disparity = False
        min_depth = 0.0
        max_depth = 0.4
        # min_disparity = 1.0 / max_depth
        # max_disparity = 1.0 / min_depth
        r_value = float(pixel1[0])
        g_value = float(pixel1[1])
        b_value = float(pixel1[2])
        depthValue = 0
        if (b_value + g_value + r_value) < 255:
            hue_value = 0
        elif (r_value >= g_value and r_value >= b_value):
            if (g_value >= b_value):
                hue_value = g_value - b_value
            else:
                hue_value = (g_value - b_value) + 1529
        elif (g_value >= r_value and g_value >= b_value):
            hue_value = b_value - r_value + 510

        elif (b_value >= g_value and b_value >= r_value):
            hue_value = r_value - g_value + 1020

        if (hue_value > 0):
            if not is_disparity:
                z_value = ((min_depth + (max_depth - min_depth) * hue_value / 1529.0))  # + 0.5);
                depthValue = (((max_depth - min_depth) * (hue_value / 1529.0)))
            else:
                pass
                # disp_value = min_disparity + (max_disparity - min_disparity) * hue_value / 1529.0
                # depthValue = ((1.0 / disp_value) / 1000 + 0.5)
        else:
            depthValue = 0
        return depthValue

    def removeColorizing(self, bbox, imdata):
        # if self.cameraView == "RIGHT":
        #     imdata = cv2.flip(imdata, 0)
        # bboxImdata = imdata[int(bbox["ymin"]):int(bbox["ymax"]),
        #              int(bbox["xmin"]):int(bbox["xmax"])]
        # shape = bboxImdata.shape
        # self.depthImage = numpy.array([[self.convertRGBtoD(j) for j in bboxImdata[i]] for i in range(shape[0])])
        shape = imdata.shape
        self.depthImage = numpy.array([[self.convertRGBtoD(j) for j in imdata[i]] for i in range(shape[0])])

    def getDepthImage(self, bbox):
        originalImData = self.getVtkImageDataAsOpenCVMat(self.depthNode)
        imdata = originalImData.copy()
        self.imgShape = imdata.shape
        shape = imdata.shape
        self.depthImage = imdata
        #
        # if len(shape) > 2:
        #     self.removeColorizing(bbox, imdata)
        # else:
        #     if self.cameraView == "RIGHT":
        #         imdata = cv2.flip(imdata, 0)
        #     bboxImdata = imdata[int(bbox["ymin"]):int(bbox["ymax"]),
        #                  int(bbox["xmin"]):int(bbox["xmax"])]
        #     self.depthImage = numpy.array([[j for j in bboxImdata[i]] for i in range(shape[0])])

    def convertDepthToPoints(self, bbox, mask):
        try:
            self.fiducialNode = slicer.util.getNode("depthFiducials")
            self.fiducialNode.RemoveAllMarkups()
            if bbox["class"] == "phantom":
                self.referenceFiducialNode = slicer.util.getNode("referenceFiducials")
                self.referenceFiducialNode.RemoveAllControlPoints()
        except slicer.util.MRMLNodeNotFoundException:
            self.fiducialNode = slicer.vtkMRMLMarkupsFiducialNode()
            self.fiducialNode.SetName("depthFiducials")
            slicer.mrmlScene.AddNode(self.fiducialNode)
            # self.fiducialNode.SetAndObserveTransformNodeID(self.initialDepthToRAS.GetID())
        if bbox["class"] == "phantom":
            try:
                self.referenceFiducialNode = slicer.util.getNode("referenceFiducials")
                self.referenceFiducialNode.RemoveAllControlPoints()
            except slicer.util.MRMLNodeNotFoundException:
                self.referenceFiducialNode = slicer.vtkMRMLMarkupsFiducialNode()
                self.referenceFiducialNode.SetName("referenceFiducials")
                slicer.mrmlScene.AddNode(self.referenceFiducialNode)
        imageShape = self.depthImage.shape
        fidAddedCount = 0
        for y in range(0, imageShape[0], 20):
            for x in range(0, imageShape[1], 20):
                if mask[y][x] > 0:
                    depthValue = self.convertRGBtoD(self.depthImage[y][x])
                    (x_mm, y_mm) = self.convert_pixels_to_mm(x, y, depthValue)
                    offset = getattr(self, "abovePlaneDepth", 0.0)

                    if bbox["class"] != "phantom":
                        self.fiducialNode.AddControlPoint(
                            numpy.array([depthValue, 480 - (bbox["ymin"] + y), bbox["xmin"] + x])
                            if self.cameraView == "RIGHT" else
                            numpy.array([(bbox["ymin"] + y), -1 * depthValue, bbox["xmin"] + x])
                        )
                    else:
                        '''self.referenceFiducialNode.AddControlPoint(
                            numpy.array([depthValue, 480 - (bbox["ymin"] + y), bbox["xmin"] + x])
                            if self.cameraView == "RIGHT" else
                            numpy.array([(bbox["ymin"] + y), -1*depthValue, bbox["xmin"] + x])
                        )'''
                        if depthValue > 0:
                            self.referenceFiducialNode.AddControlPoint(
                                numpy.array([x_mm * 1000, depthValue * 1000, y_mm * 1000]))

                    fidAddedCount += 1

    def getBestComponent(self, model, transform, bbox):
        originalDepthImg = self.depthImage.copy()
        rgb_img = self.getVtkImageDataAsOpenCVMat(self.rgbNode)
        best_mask = self.predict(rgb_img, bbox)
        # self.depthImage = numpy.where(best_mask >= 1, originalDepthImg, 0)
        self.convertDepthToPoints(bbox, best_mask)


#
# RGBD_Camera_Pair_RegistrationTest
#


class RGBD_Camera_Pair_RegistrationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_RGBD_Camera_Pair_Registration1()

    def test_RGBD_Camera_Pair_Registration1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("RGBD_Camera_Pair_Registration1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = RGBD_Camera_Pair_RegistrationLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
