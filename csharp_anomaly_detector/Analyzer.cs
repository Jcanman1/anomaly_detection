using System;
using System.Linq;
using System.Collections.Generic;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace AnomalyDetector
{
    public record ReferenceModel(double[] LabMean, double[] LabStd);

    public static class Analyzer
    {
        public static (Image<Rgba32> Image, Rgba32[] Pixels)? LoadRgbImage(string path)
        {
            try
            {
                var img = Image.Load<Rgba32>(path);
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
                for (int y=0;y<H;y++) for (int x=0;x<W;x++) white[y,x] = IsWhite(pixels[y*W + x]);
                var valid = ErodeValid(~white, W, H, borderSize);
                // collect lab for valid pixels
                var lab = new List<double[]>();
                for (int y=0;y<H;y++) for (int x=0;x<W;x++) if (valid[y,x])
                {
                    var px = pixels[y*W + x];
                    lab.Add(RgbToLab(px.R, px.G, px.B));
                }
                if (lab.Count>0) allLab.AddRange(lab);
                img.Dispose();
            }
            if (allLab.Count==0) return null;
            var arr = allLab.ToArray();
            var mean = new double[3];
            var std = new double[3];
            for (int k=0;k<3;k++)
            {
                mean[k] = arr.Select(r => r[k]).Average();
                std[k] = Math.Sqrt(arr.Select(r => Math.Pow(r[k]-mean[k],2)).Average());
                if (std[k] < 1e-6) std[k] = 1e-6;
            }
            return new ReferenceModel(mean, std);
        }

        static bool[,] ErodeValid(bool[,] valid, int W, int H, int borderSize)
        {
            // 'valid' is a boolean 2D array where true=valid; perform erosion borderSize times
            var cur = (bool[,])valid.Clone();
            for (int b=0;b<borderSize;b++)
            {
                var next = new bool[H,W];
                for (int y=0;y<H;y++) for (int x=0;x<W;x++)
                {
                    if (!cur[y,x]) { next[y,x]=false; continue; }
                    bool keep = true;
                    for (int dy=-1;dy<=1 && keep;dy++) for (int dx=-1;dx<=1 && keep;dx++)
                    {
                        int nx=x+dx, ny=y+dy;
                        if (nx<0||ny<0||nx>=W||ny>=H) { keep=false; break; }
                        if (!cur[ny,nx]) { keep=false; break; }
                    }
                    next[y,x]=keep;
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
            for (int y=0;y<H;y++) for (int x=0;x<W;x++) white[y,x] = IsWhite(pixels[y*W + x]);
            var valid_region = ErodeValid(~white, W, H, borderSize);
            var border_mask = new bool[H,W];
            for (int y=0;y<H;y++) for (int x=0;x<W;x++) border_mask[y,x] = (!white[y,x]) && (!valid_region[y,x]);
            var valid_pixels = new bool[H,W];
            for (int y=0;y<H;y++) for (int x=0;x<W;x++) valid_pixels[y,x] = (!white[y,x]) && (!border_mask[y,x]);

            var anomaly = new bool[H,W];
            for (int y=0;y<H;y++) for (int x=0;x<W;x++)
            {
                if (!valid_pixels[y,x]) continue;
                var px = pixels[y*W + x];
                var lab = RgbToLab(px.R, px.G, px.B);
                double sum=0;
                for (int k=0;k<3;k++) sum += Math.Pow((lab[k]-refModel.LabMean[k])/refModel.LabStd[k], 2);
                var distance = Math.Sqrt(sum);
                if (distance > sensitivity) anomaly[y,x]=true;
            }
            // simple morphological refine: dilation then erosion
            anomaly = Dilate(anomaly, W, H, 1);
            anomaly = Erode(anomaly, W, H, 1);
            return anomaly;
        }

        static bool[,] Dilate(bool[,] mask, int W, int H, int iters)
        {
            var cur = (bool[,])mask.Clone();
            for (int it=0; it<iters; it++)
            {
                var next = (bool[,])cur.Clone();
                for (int y=0;y<H;y++) for (int x=0;x<W;x++) if (!cur[y,x])
                {
                    bool any=false;
                    for (int dy=-1;dy<=1 && !any;dy++) for (int dx=-1;dx<=1 && !any;dx++)
                    {
                        int nx=x+dx, ny=y+dy; if(nx<0||ny<0||nx>=W||ny>=H) continue;
                        if (cur[ny,nx]) any=true;
                    }
                    if (any) next[y,x]=true;
                }
                cur = next;
            }
            return cur;
        }

        static bool[,] Erode(bool[,] mask, int W, int H, int iters)
        {
            var cur = (bool[,])mask.Clone();
            for (int it=0; it<iters; it++)
            {
                var next = (bool[,])cur.Clone();
                for (int y=0;y<H;y++) for (int x=0;x<W;x++) if (cur[y,x])
                {
                    bool keep=true;
                    for (int dy=-1;dy<=1 && keep;dy++) for (int dx=-1;dx<=1 && keep;dx++)
                    {
                        int nx=x+dx, ny=y+dy; if(nx<0||ny<0||nx>=W||ny>=H) { keep=false; break; }
                        if (!cur[ny,nx]) { keep=false; break; }
                    }
                    if (!keep) next[y,x]=false;
                }
                cur = next;
            }
            return cur;
        }

        public static Image<Rgba32> CreateHighlightedImage((Image<Rgba32> Image, Rgba32[] Pixels) imgData, bool[,] anomalyMask)
        {
            var img = imgData.Image.Clone();
            int W = img.Width, H = img.Height;
            var pixels = img.Data; // not directly accessible; use ProcessPixelRows
            img.ProcessPixelRows(accessor =>
            {
                for (int y=0;y<H;y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x=0;x<W;x++)
                    {
                        if (anomalyMask[y,x]) row[x] = new Rgba32(255,0,0,255);
                    }
                }
            });
            return img;
        }

        public static Image<Rgba32> MaskToImage(bool[,] mask)
        {
            int H = mask.GetLength(0), W = mask.GetLength(1);
            var img = new Image<Rgba32>(W, H);
            img.ProcessPixelRows(ac =>
            {
                for (int y=0;y<H;y++)
                {
                    var row = ac.GetRowSpan(y);
                    for (int x=0;x<W;x++) row[x] = mask[y,x] ? new Rgba32(255,255,255) : new Rgba32(0,0,0);
                }
            });
            return img;
        }

        // Convert 0-255 sRGB to CIE Lab (D65, standard) - float outputs
        public static double[] RgbToLab(byte r, byte g, byte b)
        {
            // sRGB 0-1
            double Rs = r/255.0, Gs = g/255.0, Bs = b/255.0;
            double R = Rs <= 0.04045 ? Rs/12.92 : Math.Pow((Rs+0.055)/1.055, 2.4);
            double G = Gs <= 0.04045 ? Gs/12.92 : Math.Pow((Gs+0.055)/1.055, 2.4);
            double B = Bs <= 0.04045 ? Bs/12.92 : Math.Pow((Bs+0.055)/1.055, 2.4);
            // Convert to XYZ (D65)
            double X = R*0.4124564 + G*0.3575761 + B*0.1804375;
            double Y = R*0.2126729 + G*0.7151522 + B*0.0721750;
            double Z = R*0.0193339 + G*0.1191920 + B*0.9503041;
            // Normalize for D65 white
            double Xn=0.95047, Yn=1.00000, Zn=1.08883;
            double xr = X/Xn, yr = Y/Yn, zr = Z/Zn;
            double eps=216.0/24389.0, kappa=24389.0/27.0;
            double fx = xr > eps ? Math.Pow(xr, 1.0/3.0) : (kappa * xr + 16) / 116.0;
            double fy = yr > eps ? Math.Pow(yr, 1.0/3.0) : (kappa * yr + 16) / 116.0;
            double fz = zr > eps ? Math.Pow(zr, 1.0/3.0) : (kappa * zr + 16) / 116.0;
            double L = 116.0 * fy - 16.0;
            double a = 500.0 * (fx - fy);
            double bb = 200.0 * (fy - fz);
            return new double[] { L, a, bb };
        }
    }
}
