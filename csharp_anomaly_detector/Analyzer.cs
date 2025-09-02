using System;
using System.Linq;
using System.Collections.Generic;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace AnomalyDetector
{
    public record ReferenceModel(double[] LabMean, double[] LabStd, double[] LabMedian, double[] LabMad);
    public record TextureFeatures(double Contrast, double Homogeneity, double Entropy, double Correlation);

    public static class Analyzer
    {
        public static (Image<Rgba32> Image, Rgba32[] Pixels)? LoadRgbImage(string path)
        {
            try
            {
                var img = SixLabors.ImageSharp.Image.Load<Rgba32>(path);
                var pixels = new Rgba32[img.Width * img.Height];
                img.CopyPixelDataTo(pixels);
                return (img, pixels);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading image {path}: {ex.Message}");
                return null;
            }
        }

        static bool IsWhite(Rgba32 p) => p.R == 255 && p.G == 255 && p.B == 255;

        public static ReferenceModel? ComputeReferenceModel(string[] refPaths, int borderSize)
        {
            var allLab = new List<double[]>();
            foreach (var p in refPaths)
            {
                var loaded = LoadRgbImage(p);
                if (loaded == null) continue;
                var (img, pixels) = loaded.Value;
                int W = img.Width, H = img.Height;
                var white = new bool[H, W];
                for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) white[y, x] = IsWhite(pixels[y * W + x]);
                var valid = ErodeValid(Not(white), W, H, borderSize);
                var lab = new List<double[]>();
                for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) if (valid[y, x])
                {
                    var px = pixels[y * W + x];
                    lab.Add(RgbToLab(px.R, px.G, px.B));
                }
                if (lab.Count > 0) allLab.AddRange(lab);
                img.Dispose();
            }
            if (allLab.Count == 0) return null;
            var arr = allLab.ToArray();
            var mean = new double[3];
            var std = new double[3];
            var median = new double[3];
            var mad = new double[3];

            for (int k = 0; k < 3; k++)
            {
                var channelValues = arr.Select(r => r[k]).OrderBy(v => v).ToArray();

                // Traditional statistics
                mean[k] = channelValues.Average();
                std[k] = Math.Sqrt(channelValues.Select(v => Math.Pow(v - mean[k], 2)).Average());
                if (std[k] < 1e-6) std[k] = 1e-6;

                // Robust statistics
                median[k] = channelValues[channelValues.Length / 2];
                var deviations = channelValues.Select(v => Math.Abs(v - median[k])).OrderBy(v => v).ToArray();
                mad[k] = deviations[deviations.Length / 2] * 1.4826; // Scale factor for normal distribution
                if (mad[k] < 1e-6) mad[k] = 1e-6;
            }
            return new ReferenceModel(mean, std, median, mad);
        }

        static bool[,] ErodeValid(bool[,] valid, int W, int H, int borderSize)
        {
            var cur = (bool[,])valid.Clone();
            for (int b = 0; b < borderSize; b++)
            {
                var next = new bool[H, W];
                for (int y = 0; y < H; y++) for (int x = 0; x < W; x++)
                {
                    if (!cur[y, x]) { next[y, x] = false; continue; }
                    bool keep = true;
                    for (int dy = -1; dy <= 1 && keep; dy++) for (int dx = -1; dx <= 1 && keep; dx++)
                    {
                        int nx = x + dx, ny = y + dy;
                        if (nx < 0 || ny < 0 || nx >= W || ny >= H) { keep = false; break; }
                        if (!cur[ny, nx]) { keep = false; break; }
                    }
                    next[y, x] = keep;
                }
                cur = next;
            }
            return cur;
        }

        public static bool[,] DetectAnomalies(ReferenceModel refModel, (Image<Rgba32> Image, Rgba32[] Pixels) imgData, double sensitivity, int borderSize)
        {
            var img = imgData.Image;
            var pixels = imgData.Pixels;
            int W = img.Width, H = img.Height;
            var white = new bool[H, W];
            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) white[y, x] = IsWhite(pixels[y * W + x]);
            var valid_region = ErodeValid(Not(white), W, H, borderSize);
            var border_mask = new bool[H, W];
            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) border_mask[y, x] = (!white[y, x]) && (!valid_region[y, x]);
            var valid_pixels = new bool[H, W];
            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) valid_pixels[y, x] = (!white[y, x]) && (!border_mask[y, x]);

            var anomaly = new bool[H, W];
            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++)
            {
                if (!valid_pixels[y, x]) continue;
                var px = pixels[y * W + x];
                var lab = RgbToLab(px.R, px.G, px.B);
                double sum = 0;
                for (int k = 0; k < 3; k++) sum += Math.Pow((lab[k] - refModel.LabMean[k]) / refModel.LabStd[k], 2);
                var distance = Math.Sqrt(sum);
                if (distance > sensitivity) anomaly[y, x] = true;
            }
            anomaly = Dilate(anomaly, W, H, 1);
            anomaly = Erode(anomaly, W, H, 1);
            return anomaly;
        }

        static bool[,] Dilate(bool[,] mask, int W, int H, int iters)
        {
            var cur = (bool[,])mask.Clone();
            for (int it = 0; it < iters; it++)
            {
                var next = (bool[,])cur.Clone();
                for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) if (!cur[y, x])
                {
                    bool any = false;
                    for (int dy = -1; dy <= 1 && !any; dy++) for (int dx = -1; dx <= 1 && !any; dx++)
                    {
                        int nx = x + dx, ny = y + dy; if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                        if (cur[ny, nx]) any = true;
                    }
                    if (any) next[y, x] = true;
                }
                cur = next;
            }
            return cur;
        }

        static bool[,] Erode(bool[,] mask, int W, int H, int iters)
        {
            var cur = (bool[,])mask.Clone();
            for (int it = 0; it < iters; it++)
            {
                var next = (bool[,])cur.Clone();
                for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) if (cur[y, x])
                {
                    bool keep = true;
                    for (int dy = -1; dy <= 1 && keep; dy++) for (int dx = -1; dx <= 1 && keep; dx++)
                    {
                        int nx = x + dx, ny = y + dy; if (nx < 0 || ny < 0 || nx >= W || ny >= H) { keep = false; break; }
                        if (!cur[ny, nx]) { keep = false; break; }
                    }
                    if (!keep) next[y, x] = false;
                }
                cur = next;
            }
            return cur;
        }

        public static Image<Rgba32> CreateHighlightedImage((Image<Rgba32> Image, Rgba32[] Pixels) imgData, bool[,] anomalyMask)
        {
            return CreateHighlightedImage(imgData, anomalyMask, null);
        }

        // Overload: accepts ellipsoid
        public static Image<Rgba32> CreateHighlightedImage((Image<Rgba32> Image, Rgba32[] Pixels) imgData, bool[,] anomalyMask, object? ellipsoidObj)
        {
            var img = imgData.Image.Clone();
            int W = img.Width, H = img.Height;
            // If ellipsoidObj is not null, extract parameters
            double[]? center = null, axes = null; double[,]? rot = null;
            string debugPath = "debug_log.txt";
            // Overwrite debug log at the start of each overlay generation
            System.IO.File.WriteAllText(debugPath, "==== New Overlay Generation ====" + System.Environment.NewLine);
            int anomalyPixelCount = 0;
            if (ellipsoidObj != null)
            {
                dynamic ellipsoid = ellipsoidObj;
                center = ellipsoid.Center;
                axes = ellipsoid.AxesLengths;
                rot = ellipsoid.Rotation;
                // Dynamically scale axes until at least 80% of anomaly pixels are inside
                if (center != null && axes != null && rot != null)
                {
                    int total = 0, bestInside = 0;
                    double bestScale = 1.0;
                    double[] scaledAxes = new double[3];
                    var anomalyLabPixels = new List<double[]>();
                    for (int y = 0; y < H; y++)
                        for (int x = 0; x < W; x++)
                            if (anomalyMask[y, x])
                            {
                                var px = imgData.Pixels[y * W + x];
                                anomalyLabPixels.Add(RgbToLab(px.R, px.G, px.B));
                            }
                    total = anomalyLabPixels.Count;
                    for (double scale = 1.0; scale < 10.0; scale += 0.05)
                    {
                        int inside = 0;
                        scaledAxes[0] = axes[0] * scale;
                        scaledAxes[1] = axes[1] * scale;
                        scaledAxes[2] = axes[2] * scale;
                        foreach (var lab in anomalyLabPixels)
                        {
                            if (IsInsideEllipsoid(lab, center, scaledAxes, rot)) inside++;
                        }
                        if (inside > bestInside) { bestInside = inside; bestScale = scale; }
                        if (inside >= 0.9 * total) break;
                    }
                    axes = new double[] { axes[0] * bestScale, axes[1] * bestScale, axes[2] * bestScale };
                    System.IO.File.AppendAllText(debugPath, $"Ellipsoid center: [{string.Join(",", center)}], axes: [{string.Join(",", axes)}], coverage: {bestInside * 100 / Math.Max(1, total)}%\n");
                }
            }
            else
            {
                System.IO.File.AppendAllText(debugPath, $"Ellipsoid is null for this image.\n");
            }
            int purpleCount = 0;
            int debugPixelCount = 0;
            int anomalyLabDebugCount = 0;
            img.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < H; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < W; x++)
                    {
                        if (anomalyMask[y, x])
                        {
                            anomalyPixelCount++;
                            var px = imgData.Pixels[y * W + x];
                            var lab = RgbToLab(px.R, px.G, px.B);
                            if (anomalyLabDebugCount < 10) {
                                System.IO.File.AppendAllText(debugPath, $"Overlay pixel ({x},{y}) LAB: [{string.Join(",", lab)}]\n");
                                anomalyLabDebugCount++;
                            }
                            if (center != null && axes != null && rot != null)
                            {
                                bool inside = IsInsideEllipsoid(lab, center, axes, rot);
                                if (debugPixelCount < 10) {
                                    System.IO.File.AppendAllText(debugPath, $"Pixel ({x},{y}) LAB: [{string.Join(",", lab)}] inside: {inside}\n");
                                    debugPixelCount++;
                                }
                                if (inside)
                                {
                                    row[x] = new Rgba32(128, 0, 128, 255); // purple
                                    purpleCount++;
                                }
                                else
                                    row[x] = new Rgba32(0, 0, 255, 255); // blue
                            }
                            else
                                row[x] = new Rgba32(0, 0, 255, 255); // blue
                        }
                    }
                }
            });
            System.IO.File.AppendAllText(debugPath, $"Anomaly overlay pixel count: {anomalyPixelCount}\n");
            System.IO.File.AppendAllText(debugPath, $"Purple overlay pixels: {purpleCount}\n");
            img.Mutate(x => x.Resize(W, H, KnownResamplers.NearestNeighbor));
            return img;
        }

        // Helper: check if LAB point is inside ellipsoid
        public static bool IsInsideEllipsoid(double[] pt, double[] center, double[] axes, double[,] rot)
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

        public static Image<Rgba32> MaskToImage(bool[,] mask)
        {
            int H = mask.GetLength(0), W = mask.GetLength(1);
            var img = new Image<Rgba32>(W, H);
            img.ProcessPixelRows(ac =>
            {
                for (int y = 0; y < H; y++)
                {
                    var row = ac.GetRowSpan(y);
                    for (int x = 0; x < W; x++)
                        row[x] = mask[y, x] ? new Rgba32(255, 255, 255) : new Rgba32(0, 0, 0);
                }
            });
            return img;
        }

        public static double[] RgbToLab(byte r, byte g, byte b)
        {
            double Rs = r / 255.0, Gs = g / 255.0, Bs = b / 255.0;
            double R = Rs <= 0.04045 ? Rs / 12.92 : Math.Pow((Rs + 0.055) / 1.055, 2.4);
            double G = Gs <= 0.04045 ? Gs / 12.92 : Math.Pow((Gs + 0.055) / 1.055, 2.4);
            double B = Bs <= 0.04045 ? Bs / 12.92 : Math.Pow((Bs + 0.055) / 1.055, 2.4);
            double X = R * 0.4124564 + G * 0.3575761 + B * 0.1804375;
            double Y = R * 0.2126729 + G * 0.7151522 + B * 0.0721750;
            double Z = R * 0.0193339 + G * 0.1191920 + B * 0.9503041;
            double Xn = 0.95047, Yn = 1.00000, Zn = 1.08883;
            double xr = X / Xn, yr = Y / Yn, zr = Z / Zn;
            double eps = 216.0 / 24389.0, kappa = 24389.0 / 27.0;
            double fx = xr > eps ? Math.Pow(xr, 1.0 / 3.0) : (kappa * xr + 16) / 116.0;
            double fy = yr > eps ? Math.Pow(yr, 1.0 / 3.0) : (kappa * yr + 16) / 116.0;
            double fz = zr > eps ? Math.Pow(zr, 1.0 / 3.0) : (kappa * zr + 16) / 116.0;
            double L = 116.0 * fy - 16.0;
            double a = 500.0 * (fx - fy);
            double bb = 200.0 * (fy - fz);
            return new double[] { L, a, bb };
        }

        static bool[,] Not(bool[,] arr)
        {
            int h = arr.GetLength(0), w = arr.GetLength(1);
            var result = new bool[h, w];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    result[y, x] = !arr[y, x];
            return result;
        }

        // ===== TEXTURE FEATURE EXTRACTION =====

        /// <summary>
        /// Extracts GLCM (Gray Level Co-occurrence Matrix) texture features
        /// </summary>
        public static TextureFeatures ExtractTextureFeatures(Image<Rgba32> image, int windowSize = 5, int distance = 1, double angle = 0)
        {
            int width = image.Width;
            int height = image.Height;

            // Convert to grayscale and quantize to 8 levels
            var grayLevels = 8;
            var glcm = new int[grayLevels, grayLevels];

            // Calculate co-occurrence matrix
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < height - distance; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    var nextRow = accessor.GetRowSpan(y + distance);

                    for (int x = 0; x < width - distance; x++)
                    {
                        // Get grayscale values (using luminance)
                        int gray1 = (row[x].R * 299 + row[x].G * 587 + row[x].B * 114) / 1000;
                        int gray2 = (nextRow[x + distance].R * 299 + nextRow[x + distance].G * 587 + nextRow[x + distance].B * 114) / 1000;

                        // Quantize to 8 levels
                        gray1 = (gray1 * (grayLevels - 1)) / 255;
                        gray2 = (gray2 * (grayLevels - 1)) / 255;

                        glcm[gray1, gray2]++;
                        glcm[gray2, gray1]++; // Symmetric matrix
                    }
                }
            });

            // Normalize GLCM
            double total = 0;
            for (int i = 0; i < grayLevels; i++)
                for (int j = 0; j < grayLevels; j++)
                    total += glcm[i, j];

            var glcmNorm = new double[grayLevels, grayLevels];
            for (int i = 0; i < grayLevels; i++)
                for (int j = 0; j < grayLevels; j++)
                    glcmNorm[i, j] = glcm[i, j] / total;

            // Calculate texture features
            double contrast = 0, homogeneity = 0, entropy = 0, correlation = 0;
            double meanI = 0, meanJ = 0, stdI = 0, stdJ = 0;

            for (int i = 0; i < grayLevels; i++)
            {
                for (int j = 0; j < grayLevels; j++)
                {
                    double p = glcmNorm[i, j];
                    if (p > 0)
                    {
                        contrast += (i - j) * (i - j) * p;
                        homogeneity += p / (1 + Math.Abs(i - j));
                        entropy -= p * Math.Log(p);
                    }
                    meanI += i * p;
                    meanJ += j * p;
                }
            }

            // Calculate standard deviations for correlation
            for (int i = 0; i < grayLevels; i++)
            {
                for (int j = 0; j < grayLevels; j++)
                {
                    double p = glcmNorm[i, j];
                    stdI += (i - meanI) * (i - meanI) * p;
                    stdJ += (j - meanJ) * (j - meanJ) * p;
                }
            }
            stdI = Math.Sqrt(stdI);
            stdJ = Math.Sqrt(stdJ);

            // Calculate correlation
            if (stdI > 0 && stdJ > 0)
            {
                for (int i = 0; i < grayLevels; i++)
                {
                    for (int j = 0; j < grayLevels; j++)
                    {
                        double p = glcmNorm[i, j];
                        correlation += ((i - meanI) * (j - meanJ) * p) / (stdI * stdJ);
                    }
                }
            }

            return new TextureFeatures(contrast, homogeneity, entropy, correlation);
        }

        /// <summary>
        /// Enhanced anomaly detection using both color and texture features
        /// </summary>
        public static bool[,] DetectAnomaliesEnhanced(
            ReferenceModel refModel,
            (Image<Rgba32> Image, Rgba32[] Pixels) imgData,
            double sensitivity,
            int borderSize,
            bool useTextureFeatures = true,
            double textureWeight = 0.3)
        {
            var img = imgData.Image;
            var pixels = imgData.Pixels;
            int W = img.Width, H = img.Height;

            var white = new bool[H, W];
            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) white[y, x] = IsWhite(pixels[y * W + x]);
            var valid_region = ErodeValid(Not(white), W, H, borderSize);
            var border_mask = new bool[H, W];
            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) border_mask[y, x] = (!white[y, x]) && (!valid_region[y, x]);
            var valid_pixels = new bool[H, W];
            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) valid_pixels[y, x] = (!white[y, x]) && (!border_mask[y, x]);

            var anomaly = new bool[H, W];

            // Extract texture features for the entire image if needed
            TextureFeatures? imgTexture = null;
            if (useTextureFeatures)
            {
                imgTexture = ExtractTextureFeatures(img);
            }

            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++)
            {
                if (!valid_pixels[y, x]) continue;

                var px = pixels[y * W + x];
                var lab = RgbToLab(px.R, px.G, px.B);

                // Color-based anomaly score (Mahalanobis distance)
                double colorDistance = 0;
                for (int k = 0; k < 3; k++)
                {
                    colorDistance += Math.Pow((lab[k] - refModel.LabMean[k]) / refModel.LabStd[k], 2);
                }
                colorDistance = Math.Sqrt(colorDistance);

                double anomalyScore = colorDistance;

                // Add texture-based anomaly if available
                if (useTextureFeatures && imgTexture != null)
                {
                    // For texture, we could compare local texture features
                    // For now, we'll use a simplified approach
                    double textureScore = 0;
                    anomalyScore = (1 - textureWeight) * colorDistance + textureWeight * textureScore;
                }

                if (anomalyScore > sensitivity) anomaly[y, x] = true;
            }

            anomaly = Dilate(anomaly, W, H, 1);
            anomaly = Erode(anomaly, W, H, 1);
            return anomaly;
        }

        /// <summary>
        /// Robust anomaly detection using Median Absolute Deviation
        /// </summary>
        public static bool[,] DetectAnomaliesRobust(
            ReferenceModel refModel,
            (Image<Rgba32> Image, Rgba32[] Pixels) imgData,
            double sensitivity,
            int borderSize,
            bool useRobustStats = true)
        {
            var img = imgData.Image;
            var pixels = imgData.Pixels;
            int W = img.Width, H = img.Height;

            var white = new bool[H, W];
            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) white[y, x] = IsWhite(pixels[y * W + x]);
            var valid_region = ErodeValid(Not(white), W, H, borderSize);
            var border_mask = new bool[H, W];
            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) border_mask[y, x] = (!white[y, x]) && (!valid_region[y, x]);
            var valid_pixels = new bool[H, W];
            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) valid_pixels[y, x] = (!white[y, x]) && (!border_mask[y, x]);

            var anomaly = new bool[H, W];
            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++)
            {
                if (!valid_pixels[y, x]) continue;
                var px = pixels[y * W + x];
                var lab = RgbToLab(px.R, px.G, px.B);
                double sum = 0;
                for (int k = 0; k < 3; k++)
                {
                    // Use robust statistics (MAD) instead of standard deviation
                    var stdToUse = useRobustStats ? refModel.LabMad[k] : refModel.LabStd[k];
                    sum += Math.Pow((lab[k] - refModel.LabMedian[k]) / stdToUse, 2);
                }
                var distance = Math.Sqrt(sum);
                if (distance > sensitivity) anomaly[y, x] = true;
            }
            anomaly = Dilate(anomaly, W, H, 1);
            anomaly = Erode(anomaly, W, H, 1);
            return anomaly;
        }
    }
}
