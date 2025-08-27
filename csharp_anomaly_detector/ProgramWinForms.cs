using System;
using System.Windows.Forms;

namespace AnomalyDetector
{
    static class ProgramWinForms
    {
        [STAThread]
        static void Main()
        {
            ApplicationConfiguration.Initialize();
            Application.Run(new MainForm());
        }
    }
}
