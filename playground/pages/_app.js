import "@/styles/globals.css";
import NavBar from "../components/NavBar";
import { ThemeProvider } from "next-themes";

export default function App({ Component, pageProps }) {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark">
      <NavBar />
      <Component {...pageProps} />
    </ThemeProvider>
  );
}