using System;
using System.Collections.Generic;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using System.Windows.Forms;

// ...existing code...
namespace AnomalyDetector
{
    public class Lab3DWindow : Form
    {
        private Ellipsoid? anomalyEllipsoid = null;
        private GLControl glControl;
        private List<double[]> refLabPoints = new List<double[]>();
        private List<double[]> anomalyLabPoints = new List<double[]>();
    private float rotX = 30, rotY = -30;
    private float zoom = 1.0f;
    private double panX = 0.0, panY = 0.0; // view pan offsets used for zoom-to-cursor
        private Point lastMouse;
        private bool dragging = false;

            public object? GetEllipsoid()
            {
                return anomalyEllipsoid;
            }

        // Helper class for MVEE result
        public class Ellipsoid
        {
            public double[] Center;
            public double[] AxesLengths; // rx, ry, rz
            public double[,] Rotation; // 3x3 rotation matrix
            public Ellipsoid(double[] center, double[] axesLengths, double[,] rotation)
            {
                Center = center;
                AxesLengths = axesLengths;
                Rotation = rotation;
            }
        }

        private Ellipsoid? ComputeMVEE(List<double[]> points, int maxIter = 100, double tol = 1e-5)
        {
            if (points.Count < 3) return null;
            int N = points.Count;
            int d = 3;
            double[,] Q = new double[d, N];
            for (int i = 0; i < N; i++)
                for (int j = 0; j < d; j++)
                    Q[j, i] = points[i][j];
            double[] u = new double[N];
            for (int i = 0; i < N; i++) u[i] = 1.0 / N;
            double[,] Xinv = new double[d, d];
            for (int iter = 0; iter < maxIter; iter++)
            {
                // Compute X = Q * diag(u) * Q^T
                double[,] X = new double[d, d];
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < d; j++)
                        for (int k = 0; k < d; k++)
                            X[j, k] += Q[j, i] * Q[k, i] * u[i];
                // Compute M = diag(Q^T * X^-1 * Q)
                Xinv = Invert3x3(X);
                double[] M = new double[N];
                for (int i = 0; i < N; i++)
                {
                    double[] qi = new double[d];
                    for (int j = 0; j < d; j++) qi[j] = Q[j, i];
                    double[] temp = MatVec(Xinv, qi);
                    double sum = 0;
                    for (int j = 0; j < d; j++) sum += qi[j] * temp[j];
                    M[i] = sum;
                }
                double maxM = M[0]; int maxIdx = 0;
                for (int i = 1; i < N; i++) if (M[i] > maxM) { maxM = M[i]; maxIdx = i; }
                double step = (maxM - d - 1) / ((d + 1) * (maxM - 1));
                if (step < tol) break;
                for (int i = 0; i < N; i++) u[i] = (1 - step) * u[i];
                u[maxIdx] += step;
            }
            // Center = Q * u
            double[] center = new double[d];
            for (int i = 0; i < N; i++)
                for (int j = 0; j < d; j++)
                    center[j] += Q[j, i] * u[i];
            // Axes = Xinv / d
            var axesMat = new double[d, d];
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    axesMat[i, j] = Xinv[i, j] / d;

            // Eigen-decompose axesMat
            var eigen = EigenDecompose(axesMat);
            double[] axesLengths = new double[3];
            for (int i = 0; i < 3; i++)
                axesLengths[i] = 1.0 / Math.Sqrt(eigen.Item1[i]); // rx, ry, rz
            double[,] rotation = eigen.Item2;
            return new Ellipsoid(center, axesLengths, rotation);
        }

            // Eigen-decomposition for symmetric 3x3 matrix
            // Returns (eigenvalues, eigenvectors as columns)
            private Tuple<double[], double[,]> EigenDecompose(double[,] mat)
            {
                double[] eigenvalues = new double[3] { 1, 1, 1 };
                double[,] eigenvectors = new double[3, 3] {
                    {1,0,0}, {0,1,0}, {0,0,1}
                };
                try
                {
                    // If MathNet.Numerics is available, use it
                    // var m = MathNet.Numerics.LinearAlgebra.Double.Matrix.Build.DenseOfArray(mat);
                    // var evd = m.Evd();
                    // for (int i = 0; i < 3; i++) eigenvalues[i] = evd.EigenValues[i].Real;
                    // for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) eigenvectors[j, i] = evd.EigenVectors[j, i].Real;
                }
                catch { }
                return Tuple.Create(eigenvalues, eigenvectors);
            }

            private double[,] Invert3x3(double[,] m)
            {
                double a = m[0,0], b = m[0,1], c = m[0,2];
                double d = m[1,0], e = m[1,1], f = m[1,2];
                double g = m[2,0], h = m[2,1], i = m[2,2];
                double A = e*i - f*h, B = f*g - d*i, C = d*h - e*g;
                double D = c*h - b*i, E = a*i - c*g, F = b*g - a*h;
                double G = b*f - c*e, H = c*d - a*f, I = a*e - b*d;
                double det = a*A + b*B + c*C;
                if (Math.Abs(det) < 1e-12) return new double[3,3];
                double[,] inv = new double[3,3];
                inv[0,0] = A/det; inv[0,1] = D/det; inv[0,2] = G/det;
                inv[1,0] = B/det; inv[1,1] = E/det; inv[1,2] = H/det;
                inv[2,0] = C/det; inv[2,1] = F/det; inv[2,2] = I/det;
                return inv;
            }
            private double[] MatVec(double[,] m, double[] v)
            {
                return new double[] {
                    m[0,0]*v[0] + m[0,1]*v[1] + m[0,2]*v[2],
                    m[1,0]*v[0] + m[1,1]*v[1] + m[1,2]*v[2],
                    m[2,0]*v[0] + m[2,1]*v[1] + m[2,2]*v[2]
                };
            }

        // Convert screen coordinates (sx,sy) to world coordinates using constructed orthographic projection
        private OpenTK.Vector3 ScreenToWorld(int sx, int sy, int W, int H, double left, double right, double bottom, double top, double near, double far, double cx, double cy, double cz)
        {
            // Build projection and modelview matrices consistent with Paint
            var proj = Matrix4.CreateOrthographicOffCenter((float)left, (float)right, (float)bottom, (float)top, (float)near, (float)far);
            // modelview: translate(-cx,-cy,-cz) * rotX * rotY * translate(cx,cy,cz)
            var mv = Matrix4.CreateTranslation((float)-cx, (float)-cy, (float)-cz)
                     * Matrix4.CreateRotationX((float)(rotX * Math.PI / 180.0))
                     * Matrix4.CreateRotationY((float)(rotY * Math.PI / 180.0))
                     * Matrix4.CreateTranslation((float)cx, (float)cy, (float)cz);

            var inv = (proj * mv).Inverted();
            // normalized device coords
            float nx = 2.0f * sx / (float)W - 1.0f;
            float ny = 1.0f - 2.0f * sy / (float)H;
            float nz = 0.0f; // middle of the depth range for ortho
            var ndc = new Vector4(nx, ny, nz, 1.0f);
            var world4 = inv * ndc;
            if (Math.Abs(world4.W) > 1e-6f) return new OpenTK.Vector3(world4.X / world4.W, world4.Y / world4.W, world4.Z / world4.W);
            return new OpenTK.Vector3(world4.X, world4.Y, world4.Z);
        }

            public Lab3DWindow()
            {
                this.Text = "3D LAB Color Visualization";
                this.Width = 900;
                this.Height = 700;
                glControl = new GLControl(new GraphicsMode(32, 24, 0, 8));
                glControl.Dock = DockStyle.Fill;
                this.Controls.Add(glControl);
                glControl.Load += GlControl_Load;
                glControl.Paint += GlControl_Paint;
                glControl.MouseDown += GlControl_MouseDown;
                glControl.MouseUp += GlControl_MouseUp;
                glControl.MouseMove += GlControl_MouseMove;
                glControl.MouseWheel += GlControl_MouseWheel;
            }
            // All helper methods and event handlers should follow here, inside the class


    private void GlControl_MouseWheel(object? sender, MouseEventArgs e)
        {
            // Compute current ortho bounds (same logic as in Paint) before changing zoom
            int W = glControl.Width, H = glControl.Height;
            if (W <= 0 || H <= 0) return;
            // compute data bounds
            double minX = double.MaxValue, minY = double.MaxValue, minZ = double.MaxValue;
            double maxX = double.MinValue, maxY = double.MinValue, maxZ = double.MinValue;
            int pts = 0;
            foreach (var p in refLabPoints) { minX = Math.Min(minX, p[0]); minY = Math.Min(minY, p[1]); minZ = Math.Min(minZ, p[2]); maxX = Math.Max(maxX, p[0]); maxY = Math.Max(maxY, p[1]); maxZ = Math.Max(maxZ, p[2]); pts++; }
            foreach (var p in anomalyLabPoints) { minX = Math.Min(minX, p[0]); minY = Math.Min(minY, p[1]); minZ = Math.Min(minZ, p[2]); maxX = Math.Max(maxX, p[0]); maxY = Math.Max(maxY, p[1]); maxZ = Math.Max(maxZ, p[2]); pts++; }

            double cxOld=0, cyOld=0, czOld=0;
            double leftOld, rightOld, bottomOld, topOld, nearOld, farOld;
            if (pts == 0)
            {
                double span = 100.0 * zoom;
                leftOld = -span; rightOld = span; bottomOld = -span; topOld = span; nearOld = -span; farOld = span;
                cxOld = 0; cyOld = 0; czOld = 0;
            }
            else
            {
                cxOld = (minX + maxX) / 2.0 + panX; cyOld = (minY + maxY) / 2.0 + panY; czOld = (minZ + maxZ) / 2.0;
                double spanX = (maxX - minX) / 2.0; double spanY = (maxY - minY) / 2.0; double spanZ = (maxZ - minZ) / 2.0;
                double span = Math.Max(Math.Max(spanX, spanY), spanZ);
                if (span < 1.0) span = 10.0;
                double margin = span * 0.25;
                double final = (span + margin) * zoom;
                leftOld = cxOld - final; rightOld = cxOld + final; bottomOld = cyOld - final; topOld = cyOld + final; nearOld = czOld - final * 2.0; farOld = czOld + final * 2.0;
            }
            // old world coords under cursor (unproject using current MV/P)
            Vector3 worldOld = ScreenToWorld(e.X, e.Y, W, H, leftOld, rightOld, bottomOld, topOld, nearOld, farOld, cxOld, cyOld, czOld);

            // apply zoom
            float oldZoom = zoom;
            if (e.Delta > 0) zoom *= 1.1f; else if (e.Delta < 0) zoom /= 1.1f;

            // recompute new bounds with updated zoom
            double leftNew, rightNew, bottomNew, topNew, nearNew, farNew;
            double cxNew=0, cyNew=0, czNew=0;
            if (pts == 0)
            {
                double span = 100.0 * zoom;
                leftNew = -span; rightNew = span; bottomNew = -span; topNew = span; nearNew = -span; farNew = span;
                cxNew = 0; cyNew = 0; czNew = 0;
            }
            else
            {
                cxNew = (minX + maxX) / 2.0 + panX; cyNew = (minY + maxY) / 2.0 + panY; czNew = (minZ + maxZ) / 2.0;
                double spanX = (maxX - minX) / 2.0; double spanY = (maxY - minY) / 2.0; double spanZ = (maxZ - minZ) / 2.0;
                double span = Math.Max(Math.Max(spanX, spanY), spanZ);
                if (span < 1.0) span = 10.0;
                double margin = span * 0.25;
                double final = (span + margin) * zoom;
                leftNew = cxNew - final; rightNew = cxNew + final; bottomNew = cyNew - final; topNew = cyNew + final; nearNew = czNew - final * 2.0; farNew = czNew + final * 2.0;
            }
            Vector3 worldNew = ScreenToWorld(e.X, e.Y, W, H, leftNew, rightNew, bottomNew, topNew, nearNew, farNew, cxNew, cyNew, czNew);

            // Adjust pan so the world point under cursor remains fixed
            panX += worldOld.X - worldNew.X;
            panY += worldOld.Y - worldNew.Y;

            glControl.Invalidate();
}
    private void GlControl_MouseDown(object? sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                dragging = true;
                lastMouse = e.Location;
            }
        }

    private void GlControl_MouseUp(object? sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
                dragging = false;
        }

    private void GlControl_MouseMove(object? sender, MouseEventArgs e)
        {
            if (dragging)
            {
                rotY += (e.X - lastMouse.X);
                rotX += (e.Y - lastMouse.Y);
                lastMouse = e.Location;
                glControl.Invalidate();
            }
        }

        public void UpdateLabPoints(List<double[]> refLab, List<double[]> anomalyLab)
        {
            if (refLab.Count == 0 && anomalyLab.Count == 0)
            {
                refLabPoints.Clear();
                anomalyLabPoints.Clear();
                anomalyEllipsoid = null;
            }
            else
            {
                refLabPoints = refLab;
                anomalyLabPoints = anomalyLab;
                // Outlier filtering using Mahalanobis distance
                var filtered = FilterOutliersMahalanobis(anomalyLabPoints);
                var ellipsoid = ComputeMVEE(filtered);
                // Dynamically scale axes until at least 90% of anomaly points are inside
                if (ellipsoid != null)
                {
                    double[] originalAxes = ellipsoid.AxesLengths;
                    double[] scaledAxes = new double[3];
                    int total = filtered.Count;
                    int bestInside = 0;
                    double bestScale = 1.0;
                    for (double scale = 1.0; scale < 10.0; scale += 0.05)
                    {
                        int inside = 0;
                        scaledAxes[0] = originalAxes[0] * scale;
                        scaledAxes[1] = originalAxes[1] * scale;
                        scaledAxes[2] = originalAxes[2] * scale;
                        foreach (var pt in filtered)
                        {
                            // Use IsInside from visualization logic
                            double[] v = new double[3] { pt[0] - ellipsoid.Center[0], pt[1] - ellipsoid.Center[1], pt[2] - ellipsoid.Center[2] };
                            double[] vLocal = new double[3];
                            for (int k = 0; k < 3; k++)
                                vLocal[k] = ellipsoid.Rotation[k,0]*v[0] + ellipsoid.Rotation[k,1]*v[1] + ellipsoid.Rotation[k,2]*v[2];
                            double val = (vLocal[0] / scaledAxes[0]) * (vLocal[0] / scaledAxes[0]) +
                                         (vLocal[1] / scaledAxes[1]) * (vLocal[1] / scaledAxes[1]) +
                                         (vLocal[2] / scaledAxes[2]) * (vLocal[2] / scaledAxes[2]);
                            if (val <= 1.0) inside++;
                        }
                        if (inside > bestInside) { bestInside = inside; bestScale = scale; }
                        if (inside >= 0.9 * total) break;
                    }
                    ellipsoid.AxesLengths = new double[] { originalAxes[0] * bestScale, originalAxes[1] * bestScale, originalAxes[2] * bestScale };
                }
                anomalyEllipsoid = ellipsoid;
            }
            glControl.Invalidate();
        }
        // Mahalanobis outlier filtering (returns filtered list)
        private List<double[]> FilterOutliersMahalanobis(List<double[]> points)
        {
            if (points.Count < 3) return points;
            int d = 3;
            // Compute centroid
            double[] mean = new double[d];
            foreach (var pt in points)
                for (int k = 0; k < d; k++)
                    mean[k] += pt[k];
            for (int k = 0; k < d; k++) mean[k] /= points.Count;
            // Compute covariance
            double[,] cov = new double[d, d];
            foreach (var pt in points)
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                        cov[i, j] += (pt[i] - mean[i]) * (pt[j] - mean[j]);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    cov[i, j] /= (points.Count - 1);
            // Invert covariance
            double[,] covInv = Invert3x3(cov);
            // Mahalanobis distance for each point
            List<double[]> filtered = new List<double[]>();
            double threshold = 4.1; // ~chi2.ppf(0.75, 3) ~ 4.1
            foreach (var pt in points)
            {
                double[] delta = new double[d];
                for (int k = 0; k < d; k++) delta[k] = pt[k] - mean[k];
                double[] temp = MatVec(covInv, delta);
                double dist = 0;
                for (int k = 0; k < d; k++) dist += delta[k] * temp[k];
                if (dist <= threshold) filtered.Add(pt);
            }
            // If too few remain, fallback to all points
            if (filtered.Count < 3) return points;
            return filtered;
        }

    private void GlControl_Load(object? sender, EventArgs e)
        {
            GL.ClearColor(0f, 0f, 0f, 1f);
            GL.Enable(EnableCap.DepthTest);
        }

    private void GlControl_Paint(object? sender, PaintEventArgs e)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            GL.MatrixMode(MatrixMode.Projection);
            GL.LoadIdentity();
            // Compute bounds from reference + anomaly points so the projection is centered on the data
            double cx = 0, cy = 0, cz = 0;
            double minX = double.MaxValue, minY = double.MaxValue, minZ = double.MaxValue;
            double maxX = double.MinValue, maxY = double.MinValue, maxZ = double.MinValue;
            int pts = 0;
            foreach (var p in refLabPoints) { minX = Math.Min(minX, p[0]); minY = Math.Min(minY, p[1]); minZ = Math.Min(minZ, p[2]); maxX = Math.Max(maxX, p[0]); maxY = Math.Max(maxY, p[1]); maxZ = Math.Max(maxZ, p[2]); pts++; }
            foreach (var p in anomalyLabPoints) { minX = Math.Min(minX, p[0]); minY = Math.Min(minY, p[1]); minZ = Math.Min(minZ, p[2]); maxX = Math.Max(maxX, p[0]); maxY = Math.Max(maxY, p[1]); maxZ = Math.Max(maxZ, p[2]); pts++; }
            double left, right, bottom, top, near, far;
            if (pts == 0)
            {
                // default symmetric view
                double span = 100.0 * zoom;
                left = -span; right = span; bottom = -span; top = span; near = -span; far = span;
            }
            else
            {
                cx = (minX + maxX) / 2.0; cy = (minY + maxY) / 2.0; cz = (minZ + maxZ) / 2.0;
                double spanX = (maxX - minX) / 2.0; double spanY = (maxY - minY) / 2.0; double spanZ = (maxZ - minZ) / 2.0;
                double span = Math.Max(Math.Max(spanX, spanY), spanZ);
                if (span < 1.0) span = 10.0; // prevent too tight framing
                double margin = span * 0.25; // add 25% margin
                double final = (span + margin) * zoom;
                left = cx - final; right = cx + final; bottom = cy - final; top = cy + final; near = cz - final * 2.0; far = cz + final * 2.0;
            }
            GL.Ortho(left, right, bottom, top, near, far);
            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadIdentity();
            // Translate so rotations occur around the data centroid
            GL.Translate((float)-cx, (float)-cy, (float)-cz);
            GL.Rotate(rotX, 1, 0, 0);
            GL.Rotate(rotY, 0, 1, 0);
            GL.Translate((float)cx, (float)cy, (float)cz);

            // Draw reference LAB points (green)
            GL.PointSize(4.0f);
            GL.Begin(PrimitiveType.Points);
            GL.Color3(0f, 1f, 0f);
            foreach (var lab in refLabPoints)
            {
                GL.Vertex3(lab[0], lab[1], lab[2]);
            }
            GL.End();

                // Draw anomaly LAB points (red)
                GL.PointSize(4.0f);
                GL.Begin(PrimitiveType.Points);
                GL.Color3(1f, 0f, 0f);
                foreach (var lab in anomalyLabPoints)
                {
                    GL.Vertex3(lab[0], lab[1], lab[2]);
                }
                GL.End();

            // Draw shaded ellipsoid around anomaly points
            if (anomalyLabPoints.Count < 3)
            {
                // Not enough points for ellipsoid
                GL.Color3(1f, 1f, 0f);
                GL.Begin(PrimitiveType.Lines);
                GL.Vertex3(0, -10, 0); GL.Vertex3(0, 10, 0);
                GL.Vertex3(-10, 0, 0); GL.Vertex3(10, 0, 0);
                GL.Vertex3(0, 0, -10); GL.Vertex3(0, 0, 10);
                GL.End();
            }
            else if (anomalyEllipsoid != null)
            {
                // Shaded ellipsoid (less transparent)
                GL.Enable(EnableCap.Blend);
                GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                    GL.Color4(0f, 0f, 1f, 0.6f);
                int stacks = 24, slices = 24;
                // Dynamically scale axes until at least 80% of anomaly points are inside
                double scale = 1.0;
                int total = anomalyLabPoints.Count;
                int inside = 0;
                double[] scaledAxes = new double[3];
                // Helper: check if a point is inside the ellipsoid
                bool IsInside(double[] pt, double[] center, double[] axes, double[,] rot)
                {
                    // Transform pt to ellipsoid local coordinates: v_local = R^T * (pt - center)
                    double[] v = new double[3] { pt[0] - center[0], pt[1] - center[1], pt[2] - center[2] };
                    double[] vLocal = new double[3];
                    for (int k = 0; k < 3; k++)
                        vLocal[k] = rot[k,0]*v[0] + rot[k,1]*v[1] + rot[k,2]*v[2]; // R^T * v
                    double val = (vLocal[0] / axes[0]) * (vLocal[0] / axes[0]) +
                                 (vLocal[1] / axes[1]) * (vLocal[1] / axes[1]) +
                                 (vLocal[2] / axes[2]) * (vLocal[2] / axes[2]);
                    return val <= 1.0;
                }
                double bestScale = 1.0;
                int bestInside = 0;
                for (scale = 1.0; scale < 10.0; scale += 0.05)
                {
                    inside = 0;
                    scaledAxes[0] = anomalyEllipsoid.AxesLengths[0] * scale;
                    scaledAxes[1] = anomalyEllipsoid.AxesLengths[1] * scale;
                    scaledAxes[2] = anomalyEllipsoid.AxesLengths[2] * scale;
                    foreach (var pt in anomalyLabPoints)
                        if (IsInside(pt, anomalyEllipsoid.Center, scaledAxes, anomalyEllipsoid.Rotation)) inside++;
                    if (inside > bestInside) { bestInside = inside; bestScale = scale; }
                    if (inside >= 0.9 * total) break;
                }
                scaledAxes[0] = anomalyEllipsoid.AxesLengths[0] * bestScale;
                scaledAxes[1] = anomalyEllipsoid.AxesLengths[1] * bestScale;
                scaledAxes[2] = anomalyEllipsoid.AxesLengths[2] * bestScale;
                // Show coverage percentage in window title
                this.Text = $"3D LAB Color Visualization - Ellipsoid coverage: {bestInside * 100 / Math.Max(1, total)}%";
                for (int i = 0; i < stacks; i++)
                {
                    double phi1 = Math.PI * i / stacks;
                    double phi2 = Math.PI * (i+1) / stacks;
                    GL.Begin(PrimitiveType.QuadStrip);
                    for (int j = 0; j <= slices; j++)
                    {
                        double theta = 2 * Math.PI * j / slices;
                        foreach (var phi in new[]{phi1, phi2})
                        {
                            double x = Math.Cos(theta) * Math.Sin(phi);
                            double y = Math.Sin(theta) * Math.Sin(phi);
                            double z = Math.Cos(phi);
                            double[] v = new double[3] { x * scaledAxes[0], y * scaledAxes[1], z * scaledAxes[2] };
                            double[] vRot = new double[3];
                            for (int k = 0; k < 3; k++)
                                vRot[k] = anomalyEllipsoid.Rotation[k,0]*v[0] + anomalyEllipsoid.Rotation[k,1]*v[1] + anomalyEllipsoid.Rotation[k,2]*v[2];
                            double[] p = new double[3] {
                                anomalyEllipsoid.Center[0] + vRot[0],
                                anomalyEllipsoid.Center[1] + vRot[1],
                                anomalyEllipsoid.Center[2] + vRot[2]
                            };
                            GL.Vertex3(p[0], p[1], p[2]);
                        }
                    }
                    GL.End();
                }
                for (int i = 0; i <= stacks; i += 2)
                {
                    GL.Begin(PrimitiveType.LineStrip);
                    double phi = Math.PI * i / stacks;
                    for (int j = 0; j <= slices; j++)
                    {
                        double theta = 2 * Math.PI * j / slices;
                        double x = Math.Cos(theta) * Math.Sin(phi);
                        double y = Math.Sin(theta) * Math.Sin(phi);
                        double z = Math.Cos(phi);
                        double[] v = new double[3] { x * scaledAxes[0], y * scaledAxes[1], z * scaledAxes[2] };
                        double[] vRot = new double[3];
                        for (int k = 0; k < 3; k++)
                            vRot[k] = anomalyEllipsoid.Rotation[k,0]*v[0] + anomalyEllipsoid.Rotation[k,1]*v[1] + anomalyEllipsoid.Rotation[k,2]*v[2];
                        double[] p = new double[3] {
                            anomalyEllipsoid.Center[0] + vRot[0],
                            anomalyEllipsoid.Center[1] + vRot[1],
                            anomalyEllipsoid.Center[2] + vRot[2]
                        };
                        GL.Vertex3(p[0], p[1], p[2]);
                    }
                    GL.End();
                }
                glControl.SwapBuffers();
            }
        }

    }
}
