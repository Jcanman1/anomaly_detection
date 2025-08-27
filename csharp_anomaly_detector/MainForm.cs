using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;

namespace AnomalyDetector
{
    public class MainForm : Form
    {
        private Button btnLoadRef;
        private Button btnLoadInput;
        private Label lblRefCount;
        private Label lblInputCount;
        private NumericUpDown numSensitivity;
        private NumericUpDown numBorder;
        private Button btnAnalyze;
        private FlowLayoutPanel flowResults;

        private List<string> refFiles = new List<string>();
        private List<string> inputFiles = new List<string>();

        public MainForm()
        {
            this.Text = "Anomaly Detector - WinForms GUI";
            this.Width = 900;
            this.Height = 700;

            btnLoadRef = new Button() { Text = "Load Reference (Sample 1)", Left = 10, Top = 10, Width = 180 };
            btnLoadRef.Click += BtnLoadRef_Click;
            this.Controls.Add(btnLoadRef);

            lblRefCount = new Label() { Left = 200, Top = 15, Width = 300, Text = "0 files" };
            this.Controls.Add(lblRefCount);

            btnLoadInput = new Button() { Text = "Load Input Samples", Left = 10, Top = 50, Width = 180 };
            btnLoadInput.Click += BtnLoadInput_Click;
            this.Controls.Add(btnLoadInput);

            lblInputCount = new Label() { Left = 200, Top = 55, Width = 300, Text = "0 files" };
            this.Controls.Add(lblInputCount);

            var lblSens = new Label() { Text = "Sensitivity:", Left = 10, Top = 95, Width = 80 };
            this.Controls.Add(lblSens);
            numSensitivity = new NumericUpDown() { Left = 100, Top = 90, Width = 80, DecimalPlaces = 2, Increment = 0.05M, Minimum = 0.5M, Maximum = 20M, Value = 2.50M };
            this.Controls.Add(numSensitivity);

            var lblBorder = new Label() { Text = "Border erosion:", Left = 200, Top = 95, Width = 90 };
            this.Controls.Add(lblBorder);
            numBorder = new NumericUpDown() { Left = 300, Top = 90, Width = 60, Minimum = 0, Maximum = 10, Value = 1 };
            this.Controls.Add(numBorder);

            btnAnalyze = new Button() { Text = "Analyze", Left = 10, Top = 130, Width = 120, Height = 30 };
            btnAnalyze.Click += BtnAnalyze_Click;
            this.Controls.Add(btnAnalyze);

            flowResults = new FlowLayoutPanel() { Left = 10, Top = 180, Width = this.ClientSize.Width - 20, Height = this.ClientSize.Height - 200, AutoScroll = true };
            flowResults.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right;
            this.Controls.Add(flowResults);
        }

        private void BtnLoadRef_Click(object? sender, EventArgs e)
        {
            using var dlg = new OpenFileDialog();
            dlg.Multiselect = true;
            dlg.Filter = "Image Files|*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp";
            if (dlg.ShowDialog() == DialogResult.OK)
            {
                refFiles.Clear();
                refFiles.AddRange(dlg.FileNames);
                lblRefCount.Text = $"{refFiles.Count} files";
            }
        }

        private void BtnLoadInput_Click(object? sender, EventArgs e)
        {
            using var dlg = new OpenFileDialog();
            dlg.Multiselect = true;
            dlg.Filter = "Image Files|*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp";
            if (dlg.ShowDialog() == DialogResult.OK)
            {
                inputFiles.Clear();
                inputFiles.AddRange(dlg.FileNames);
                lblInputCount.Text = $"{inputFiles.Count} files";
            }
        }

        private async void BtnAnalyze_Click(object? sender, EventArgs e)
        {
            if (refFiles.Count == 0)
            {
                MessageBox.Show(this, "Please load at least one reference image (Sample 1)", "Missing Reference", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            if (inputFiles.Count == 0)
            {
                MessageBox.Show(this, "Please load at least one input image to analyze", "Missing Input", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            btnAnalyze.Enabled = false;
            flowResults.Controls.Clear();

            var model = Analyzer.ComputeReferenceModel(refFiles.ToArray(), (int)numBorder.Value);
            if (model == null)
            {
                MessageBox.Show(this, "Failed to compute reference model from loaded reference images.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                btnAnalyze.Enabled = true;
                return;
            }

            try
            {
                foreach (var input in inputFiles)
                {
                    var loaded = Analyzer.LoadRgbImage(input);
                    if (loaded == null) continue;
                    var mask = Analyzer.DetectAnomalies(model, loaded.Value, (double)numSensitivity.Value, (int)numBorder.Value);
                    var highlighted = Analyzer.CreateHighlightedImage(loaded.Value, mask);

                    // Convert ImageSharp image to System.Drawing.Bitmap via memory stream
                    using var ms = new MemoryStream();
                    highlighted.Save(ms, new PngEncoder());
                    ms.Seek(0, SeekOrigin.Begin);
                    var bmp = new System.Drawing.Bitmap(ms);

                    var pic = new PictureBox();
                    pic.Width = 220; pic.Height = 220; pic.SizeMode = PictureBoxSizeMode.Zoom;
                    pic.Image = new System.Drawing.Bitmap(bmp);

                    var panel = new Panel() { Width = 240, Height = 260, BorderStyle = BorderStyle.FixedSingle };
                    var lbl = new Label() { Text = Path.GetFileName(input), Dock = DockStyle.Bottom, Height = 40, TextAlign = System.Drawing.ContentAlignment.MiddleCenter };
                    panel.Controls.Add(pic);
                    pic.Dock = DockStyle.Top;
                    panel.Controls.Add(lbl);
                    flowResults.Controls.Add(panel);

                    // Also free bitmap
                    bmp.Dispose();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, "Error during analysis: " + ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            btnAnalyze.Enabled = true;
        }
    }
}
