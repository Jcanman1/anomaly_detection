using System;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Collections.Generic;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace AnomalyDetector
{
    class Program
    {
        static void PrintUsage()
        {
            Console.WriteLine("Usage:\n  dotnet run -- --ref <refPaths(semi-colon)> --input <inputPaths(semi-colon)> [--sensitivity 2.5] [--border 1] [--outdir out]");
        }

        static int Main(string[] args)
        {
            if (args.Length == 0)
            {
                PrintUsage();
                return 1;
            }

            string refs = null!;
            string inputs = null!;
            double sensitivity = 2.5;
            int border = 1;
            string outdir = Path.Combine(Directory.GetCurrentDirectory(), "out");

            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--ref": refs = args[++i]; break;
                    case "--input": inputs = args[++i]; break;
                    case "--sensitivity": sensitivity = double.Parse(args[++i]); break;
                    case "--border": border = int.Parse(args[++i]); break;
                    case "--outdir": outdir = args[++i]; break;
                    case "--help":
                    case "-h":
                        PrintUsage();
                        return 0;
                }
            }

            if (string.IsNullOrEmpty(refs) || string.IsNullOrEmpty(inputs))
            {
                PrintUsage();
                return 1;
            }

            Directory.CreateDirectory(outdir);

            var refPaths = refs.Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries).Select(p => p.Trim()).ToArray();
            var inputPaths = inputs.Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries).Select(p => p.Trim()).ToArray();

            Console.WriteLine($"Loading {refPaths.Length} reference image(s) and {inputPaths.Length} input image(s)");

            var refModel = Analyzer.ComputeReferenceModel(refPaths, border);
            if (refModel == null)
            {
                Console.WriteLine("Failed to compute reference model. Ensure reference images have valid non-white pixels.");
                return 1;
            }

            Console.WriteLine($"Reference LAB mean: [{string.Join(", ", refModel.LabMean.Select(v => v.ToString("F2")))}]");
            Console.WriteLine($"Reference LAB std : [{string.Join(", ", refModel.LabStd.Select(v => v.ToString("F2")))}]");

            int idx = 0;
            foreach (var input in inputPaths)
            {
                idx++;
                Console.WriteLine($"Processing: {input}");
                var img = Analyzer.LoadRgbImage(input);
                if (img == null)
                {
                    Console.WriteLine($"  Could not load {input}");
                    continue;
                }

                var mask = Analyzer.DetectAnomalies(refModel, img.Value, sensitivity, border);
                var highlighted = Analyzer.CreateHighlightedImage(img.Value, mask);
                var outPath = Path.Combine(outdir, Path.GetFileNameWithoutExtension(input) + "_highlighted.png");
                highlighted.Save(outPath);
                Console.WriteLine($"  Saved highlighted image to {outPath}");

                var maskPath = Path.Combine(outdir, Path.GetFileNameWithoutExtension(input) + "_mask.png");
                using (var mimg = Analyzer.MaskToImage(mask)) mimg.Save(maskPath);
                Console.WriteLine($"  Saved mask image to {maskPath}");
            }

            Console.WriteLine("Done.");
            return 0;
        }
    }
}
