using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;

namespace AnomalyDetector
{
    public class MainForm : Form
    {
        private Button btnLoadRef;
        private Button btnClearRef;
        private Button btnLoadInput;
        private Button btnClearInput;
        private Label lblRefCount;
        private Label lblInputCount;
        private NumericUpDown numSensitivity;
        private NumericUpDown numBorder;
    // Removed btnAnalyze
    // Removed flowResults
        private CheckBox chkRobustStats;
        private CheckBox chkTextureFeatures;
        private NumericUpDown numTextureWeight;

        private List<string> refFiles = new List<string>();
        private List<string> inputFiles = new List<string>();

        private FlowLayoutPanel refThumbPanel;
        private FlowLayoutPanel inputThumbPanel;

        public MainForm()
            // Attach event handlers for live update

        {
            this.Text = "Anomaly Detector - WinForms GUI";
            this.Width = 2000;
            this.Height = 1200;
            this.AutoScaleMode = AutoScaleMode.Font;

            var mainLayout = new TableLayoutPanel {
                Dock = DockStyle.Fill,
                ColumnCount = 6,
                RowCount = 4,
                AutoSize = true,
                AutoSizeMode = AutoSizeMode.GrowAndShrink,
                Padding = new Padding(10)
            };
            mainLayout.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize)); // 0
            mainLayout.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize)); // 1
            mainLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 200)); // 2
            mainLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 200)); // 3
            mainLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 200)); // 4
            mainLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100)); // 5
            mainLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 250));
            mainLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 250));
            mainLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));
            mainLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 400));

            btnLoadRef = new Button() { Text = "Load Reference (Sample 1)", AutoSize = true };
            btnLoadRef.Click += BtnLoadRef_Click;
            btnClearRef = new Button() { Text = "Clear", AutoSize = true };
            btnClearRef.Click += BtnClearRef_Click;
            var refButtonsPanel = new FlowLayoutPanel() { AutoSize = true, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
            refButtonsPanel.Controls.Add(btnLoadRef);
            refButtonsPanel.Controls.Add(btnClearRef);
            mainLayout.Controls.Add(refButtonsPanel, 0, 0);

            lblRefCount = new Label() { Text = "0 files", AutoSize = true, TextAlign = System.Drawing.ContentAlignment.MiddleLeft };
            mainLayout.Controls.Add(lblRefCount, 1, 0);

            btnLoadInput = new Button() { Text = "Load Input Samples", AutoSize = true };
            btnLoadInput.Click += BtnLoadInput_Click;
            btnClearInput = new Button() { Text = "Clear", AutoSize = true };
            btnClearInput.Click += BtnClearInput_Click;
            var inputButtonsPanel = new FlowLayoutPanel() { AutoSize = true, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
            inputButtonsPanel.Controls.Add(btnLoadInput);
            inputButtonsPanel.Controls.Add(btnClearInput);
            mainLayout.Controls.Add(inputButtonsPanel, 0, 1);

            lblInputCount = new Label() { Text = "0 files", AutoSize = true, TextAlign = System.Drawing.ContentAlignment.MiddleLeft };
            mainLayout.Controls.Add(lblInputCount, 1, 1);

            var lblSens = new Label() { Text = "Sensitivity:", AutoSize = true, TextAlign = System.Drawing.ContentAlignment.MiddleLeft };
            mainLayout.Controls.Add(lblSens, 0, 2);
            numSensitivity = new NumericUpDown() { DecimalPlaces = 2, Increment = 0.05M, Minimum = 0.5M, Maximum = 20M, Value = 2.50M, AutoSize = true };
            mainLayout.Controls.Add(numSensitivity, 1, 2);

            var lblBorder = new Label() { Text = "Border erosion:", AutoSize = true, TextAlign = System.Drawing.ContentAlignment.MiddleLeft };
            mainLayout.Controls.Add(lblBorder, 2, 2);
            numBorder = new NumericUpDown() { Minimum = 0, Maximum = 10, Value = 1, AutoSize = true };
            mainLayout.Controls.Add(numBorder, 3, 2);

            var lblTextureWeight = new Label() { Text = "Texture Weight:", AutoSize = true, TextAlign = System.Drawing.ContentAlignment.MiddleLeft };
            mainLayout.Controls.Add(lblTextureWeight, 4, 2);
            numTextureWeight = new NumericUpDown() { DecimalPlaces = 2, Increment = 0.1M, Minimum = 0M, Maximum = 1M, Value = 0.3M, AutoSize = true };
            mainLayout.Controls.Add(numTextureWeight, 5, 2);

            // Add enhanced detection options
            chkRobustStats = new CheckBox() { Text = "Use Robust Statistics (MAD)", Checked = true, AutoSize = true };
            mainLayout.Controls.Add(chkRobustStats, 0, 3);

            chkTextureFeatures = new CheckBox() { Text = "Use Texture Features", Checked = false, AutoSize = true };
            mainLayout.Controls.Add(chkTextureFeatures, 1, 3);


            // Removed flowResults from layout

            refThumbPanel = new FlowLayoutPanel() { AutoSize = false, Height = 500, Width = 2000, FlowDirection = FlowDirection.LeftToRight, WrapContents = false, AutoScroll = true };
            inputThumbPanel = new FlowLayoutPanel() { AutoSize = false, Height = 500, Width = 2000, FlowDirection = FlowDirection.LeftToRight, WrapContents = false, AutoScroll = true };
            mainLayout.Controls.Add(refThumbPanel, 2, 0);
            mainLayout.SetColumnSpan(refThumbPanel, 4);
            mainLayout.Controls.Add(inputThumbPanel, 2, 1);
            mainLayout.SetColumnSpan(inputThumbPanel, 4);

            this.Controls.Add(mainLayout);
            // Attach event handlers for live update (after controls are initialized)
            numSensitivity.ValueChanged += (s, e) => UpdateAnalysisOverlays();
            numBorder.ValueChanged += (s, e) => UpdateAnalysisOverlays();
            numTextureWeight.ValueChanged += (s, e) => UpdateAnalysisOverlays();
            chkRobustStats.CheckedChanged += (s, e) => UpdateAnalysisOverlays();
            chkTextureFeatures.CheckedChanged += (s, e) => UpdateAnalysisOverlays();
        }

        private void BtnClearRef_Click(object? sender, EventArgs e)
        {
            refFiles.Clear();
            lblRefCount.Text = "0 files";
            refThumbPanel.Controls.Clear();
            // Keep a minimum reasonable width so layout remains stable
            refThumbPanel.Width = 800;
            // Clear any analysis results since inputs changed
            refThumbPanel.Controls.Clear();
        }

        private void BtnClearInput_Click(object? sender, EventArgs e)
        {
            inputFiles.Clear();
            lblInputCount.Text = "0 files";
            inputThumbPanel.Controls.Clear();
            inputThumbPanel.Width = 800;
            inputThumbPanel.Controls.Clear();
        }

        private void UpdateAnalysisOverlays()
        {
            // Update overlays for reference images
            refThumbPanel.Controls.Clear();
            if (refFiles.Count > 0)
            {
                var model = Analyzer.ComputeReferenceModel(refFiles.ToArray(), (int)numBorder.Value);
                foreach (var refPath in refFiles)
                {
                    var loaded = Analyzer.LoadRgbImage(refPath);
                    if (loaded == null) continue;
                    bool[,] mask;
                    if (chkTextureFeatures.Checked && chkRobustStats.Checked)
                    {
                        mask = Analyzer.DetectAnomaliesEnhanced(
                            model, loaded.Value, (double)numSensitivity.Value, (int)numBorder.Value,
                            true, (double)numTextureWeight.Value);
                    }
                    else if (chkRobustStats.Checked)
                    {
                        mask = Analyzer.DetectAnomaliesRobust(
                            model, loaded.Value, (double)numSensitivity.Value, (int)numBorder.Value, true);
                    }
                    else
                    {
                        mask = Analyzer.DetectAnomalies(model, loaded.Value, (double)numSensitivity.Value, (int)numBorder.Value);
                    }
                    var highlighted = Analyzer.CreateHighlightedImage(loaded.Value, mask);
                    highlighted.Mutate(x => x.Resize(200, 200));
                    using var ms = new MemoryStream();
                    highlighted.Save(ms, new PngEncoder());
                    ms.Seek(0, SeekOrigin.Begin);
                    var bmp = new System.Drawing.Bitmap(ms);
                    var pic = new PictureBox() {
                        Width = 200,
                        Height = 200,
                        SizeMode = PictureBoxSizeMode.Zoom,
                        Image = new System.Drawing.Bitmap(bmp),
                        Margin = new Padding(5)
                    };
                    pic.Dock = DockStyle.None;
                    refThumbPanel.Controls.Add(pic);
                    bmp.Dispose();
                }
            }

            // Update overlays for input images
            inputThumbPanel.Controls.Clear();
            if (inputFiles.Count > 0 && refFiles.Count > 0)
            {
                var model = Analyzer.ComputeReferenceModel(refFiles.ToArray(), (int)numBorder.Value);
                foreach (var file in inputFiles)
                {
                    var loaded = Analyzer.LoadRgbImage(file);
                    if (loaded == null) continue;
                    bool[,] mask;
                    if (chkTextureFeatures.Checked && chkRobustStats.Checked)
                    {
                        mask = Analyzer.DetectAnomaliesEnhanced(
                            model, loaded.Value, (double)numSensitivity.Value, (int)numBorder.Value,
                            true, (double)numTextureWeight.Value);
                    }
                    else if (chkRobustStats.Checked)
                    {
                        mask = Analyzer.DetectAnomaliesRobust(
                            model, loaded.Value, (double)numSensitivity.Value, (int)numBorder.Value, true);
                    }
                    else
                    {
                        mask = Analyzer.DetectAnomalies(model, loaded.Value, (double)numSensitivity.Value, (int)numBorder.Value);
                    }
                    var highlighted = Analyzer.CreateHighlightedImage(loaded.Value, mask);
                    highlighted.Mutate(x => x.Resize(200, 200));
                    using var ms = new MemoryStream();
                    highlighted.Save(ms, new PngEncoder());
                    ms.Seek(0, SeekOrigin.Begin);
                    var bmp = new System.Drawing.Bitmap(ms);
                    var pic = new PictureBox() {
                        Width = 200,
                        Height = 200,
                        SizeMode = PictureBoxSizeMode.Zoom,
                        Image = new System.Drawing.Bitmap(bmp),
                        Margin = new Padding(5)
                    };
                    pic.Dock = DockStyle.None;
                    inputThumbPanel.Controls.Add(pic);
                    bmp.Dispose();
                }
            }
            else if (inputFiles.Count > 0)
            {
                foreach (var file in inputFiles)
                {
                    try
                    {
                        using var img = SixLabors.ImageSharp.Image.Load<Rgba32>(file);
                        img.Mutate(x => x.Resize(200, 200));
                        using var ms = new MemoryStream();
                        img.Save(ms, new PngEncoder());
                        ms.Seek(0, SeekOrigin.Begin);
                        var bmp = new System.Drawing.Bitmap(ms);
                        var pic = new PictureBox() {
                            Width = 200,
                            Height = 200,
                            SizeMode = PictureBoxSizeMode.Zoom,
                            Image = new System.Drawing.Bitmap(bmp),
                            Margin = new Padding(5)
                        };
                        pic.Dock = DockStyle.None;
                        inputThumbPanel.Controls.Add(pic);
                        bmp.Dispose();
                    }
                    catch { }
                }
            }
        }

        private void BtnLoadRef_Click(object? sender, EventArgs e)
        {
            using var dlg = new OpenFileDialog();
            dlg.Multiselect = true;
            dlg.Filter = "Image Files|*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp";
            dlg.RestoreDirectory = true;
            if (dlg.ShowDialog() == DialogResult.OK)
            {
                // Add new files without clearing existing ones
                foreach (var file in dlg.FileNames)
                {
                    if (!refFiles.Contains(file))
                    {
                        refFiles.Add(file);
                    }
                }
                
                // Calculate dynamic width based on number of images
                int imageWidth = 200; // width of each thumbnail
                int margin = 5; // margin between images
                int minWidth = 800; // minimum width
                int calculatedWidth = Math.Max(minWidth, (refFiles.Count * (imageWidth + margin)) + 50);
                
                refThumbPanel.Width = calculatedWidth;
                lblRefCount.Text = $"{refFiles.Count} files";
                
                // Update thumbnails without clearing
                refThumbPanel.Controls.Clear();
                foreach (var file in refFiles)
                {
                    try
                    {
                        using var img = SixLabors.ImageSharp.Image.Load<Rgba32>(file);
                        img.Mutate(x => x.Resize(200, 200));
                        using var ms = new MemoryStream();
                        img.Save(ms, new PngEncoder());
                        ms.Seek(0, SeekOrigin.Begin);
                        var bmp = new System.Drawing.Bitmap(ms);
                        var pic = new PictureBox() {
                            Width = 200,
                            Height = 200,
                            SizeMode = PictureBoxSizeMode.Zoom,
                            Image = new System.Drawing.Bitmap(bmp),
                            Margin = new Padding(5)
                        };
                        pic.Dock = DockStyle.None;
                        refThumbPanel.Controls.Add(pic);
                        bmp.Dispose();
                    }
                    catch { }
                }
            }
        }

        private void BtnLoadInput_Click(object? sender, EventArgs e)
        {
            using var dlg = new OpenFileDialog();
            dlg.Multiselect = true;
            dlg.Filter = "Image Files|*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp";
            dlg.RestoreDirectory = true;
            if (dlg.ShowDialog() == DialogResult.OK)
            {
                // Add new files without clearing existing ones
                foreach (var file in dlg.FileNames)
                {
                    if (!inputFiles.Contains(file))
                    {
                        inputFiles.Add(file);
                    }
                }

                // Calculate dynamic width based on number of images
                int imageWidth = 200; // width of each thumbnail
                int margin = 5; // margin between images
                int minWidth = 800; // minimum width
                int calculatedWidth = Math.Max(minWidth, (inputFiles.Count * (imageWidth + margin)) + 50);

                inputThumbPanel.Width = calculatedWidth;
                lblInputCount.Text = $"{inputFiles.Count} files";

                // Apply overlays directly to the input thumbnails at the top
                inputThumbPanel.Controls.Clear();
                // Removed flowResults.Controls.Clear();
                bool hasRef = refFiles.Count > 0;
                ReferenceModel? model = null;
                if (hasRef)
                {
                    model = Analyzer.ComputeReferenceModel(refFiles.ToArray(), (int)numBorder.Value);
                }
                foreach (var file in inputFiles)
                {
                    try
                    {
                        using var img = SixLabors.ImageSharp.Image.Load<Rgba32>(file);
                        img.Mutate(x => x.Resize(200, 200));
                        System.Drawing.Bitmap bmp;
                        if (hasRef && model is not null)
                        {
                            bool[,] mask;
                            var loaded = Analyzer.LoadRgbImage(file);
                            if (loaded == null)
                            {
                                // fallback to original image
                                using var ms = new MemoryStream();
                                img.Save(ms, new PngEncoder());
                                ms.Seek(0, SeekOrigin.Begin);
                                bmp = new System.Drawing.Bitmap(ms);
                            }
                            else
                            {
                                if (chkTextureFeatures.Checked && chkRobustStats.Checked)
                                {
                                    mask = Analyzer.DetectAnomaliesEnhanced(
                                        model, loaded.Value, (double)numSensitivity.Value, (int)numBorder.Value,
                                        true, (double)numTextureWeight.Value);
                                }
                                else if (chkRobustStats.Checked)
                                {
                                    mask = Analyzer.DetectAnomaliesRobust(
                                        model, loaded.Value, (double)numSensitivity.Value, (int)numBorder.Value, true);
                                }
                                else
                                {
                                    mask = Analyzer.DetectAnomalies(model, loaded.Value, (double)numSensitivity.Value, (int)numBorder.Value);
                                }
                                var highlighted = Analyzer.CreateHighlightedImage(loaded.Value, mask);
                                highlighted.Mutate(x => x.Resize(200, 200));
                                using var ms = new MemoryStream();
                                highlighted.Save(ms, new PngEncoder());
                                ms.Seek(0, SeekOrigin.Begin);
                                bmp = new System.Drawing.Bitmap(ms);
                            }
                        }
                        else
                        {
                            using var ms = new MemoryStream();
                            img.Save(ms, new PngEncoder());
                            ms.Seek(0, SeekOrigin.Begin);
                            bmp = new System.Drawing.Bitmap(ms);
                        }
                        var pic = new PictureBox() {
                            Width = 200,
                            Height = 200,
                            SizeMode = PictureBoxSizeMode.Zoom,
                            Image = new System.Drawing.Bitmap(bmp),
                            Margin = new Padding(5)
                        };
                        pic.Dock = DockStyle.None;
                        inputThumbPanel.Controls.Add(pic);
                        bmp.Dispose();
                    }
                    catch { }
                }
            }
        }

        private async void BtnAnalyze_Click(object? sender, EventArgs e)
        {
    // Removed BtnAnalyze_Click and all related logic
        }
    }
}
