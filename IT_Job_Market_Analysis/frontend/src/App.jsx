import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend, LineChart, Line, CartesianGrid } from 'recharts';
import { Search, MessageSquare, TrendingUp, Zap, ServerCrash, Hash, Activity } from 'lucide-react';

const API_BASE = 'http://localhost:8000';
const PIE_COLORS = { 'Positive': '#10b981', 'Negative': '#ef4444', 'Neutral': '#f59e0b' };

function App() {
  const [reviews, setReviews] = useState([]);
  const [summary, setSummary] = useState(null);
  const [nlpInsights, setNlpInsights] = useState(null);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [sentimentFilter, setSentimentFilter] = useState("");
  const [appFilter, setAppFilter] = useState("");
  
  // Live Analysis States
  const [appIdInput, setAppIdInput] = useState("");
  const [isLiveMode, setIsLiveMode] = useState(false);
  const [liveError, setLiveError] = useState("");

  useEffect(() => {
    fetchStaticData();
  }, []);

  const fetchStaticData = async () => {
    setLoading(true);
    setIsLiveMode(false);
    setLiveError("");
    try {
      const [revRes, sumRes] = await Promise.all([
        axios.get(`${API_BASE}/reviews`),
        axios.get(`${API_BASE}/sentiment/summary`)
      ]);
      setReviews(revRes.data);
      setSummary(sumRes.data);
      setNlpInsights(null); // Static data doesn't have the new NLP pipeline computed
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleLiveAnalysis = async () => {
    if (!appIdInput.trim()) {
      fetchStaticData();
      return;
    }
    
    setLoading(true);
    setIsLiveMode(true);
    setLiveError("");
    
    try {
      const res = await axios.get(`${API_BASE}/analyze-live?app_id=${encodeURIComponent(appIdInput.trim())}&limit=150`);
      if (res.data.error) {
        setLiveError(res.data.error);
        setSummary(null);
        setReviews([]);
        setNlpInsights(null);
      } else {
        setReviews(res.data.reviews);
        setSummary(res.data.summary);
        setNlpInsights(res.data.nlp_insights);
      }
    } catch (error) {
      setLiveError("Server Connection Failed. Make sure Uvicorn is active.");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const filteredReviews = reviews.filter(r => {
    const matchSearch = (r.company || "").toLowerCase().includes(searchTerm.toLowerCase()) || 
                        (r.review_text || "").toLowerCase().includes(searchTerm.toLowerCase());
    const matchSentiment = sentimentFilter === "" || r.predicted_sentiment === sentimentFilter;
    const matchApp = appFilter === "" || r.company === appFilter;
    return matchSearch && matchSentiment && matchApp;
  });

  const uniqueCompanies = [...new Set(reviews.map(r => r.company))].filter(Boolean).sort();

  const pieData = summary && summary.sentiment_distribution 
      ? Object.entries(summary.sentiment_distribution).map(([name, value]) => ({ name, value }))
      : [];
      
  const barData = summary && summary.company_polarity
      ? Object.entries(summary.company_polarity).map(([name, value]) => ({ name, value: parseFloat(value.toFixed(2)) }))
      : [];

  return (
    <div className="min-h-screen bg-slate-50 p-6 font-sans">
      <header className="mb-6 border-b pb-4 border-slate-200">
        <h1 className="text-3xl font-bold text-slate-900 flex items-center">
          Business Sentiment Dashboard
        </h1>
        <p className="text-slate-500 mt-2">AI-driven analysis of Customer Reviews via Natural Language Processing</p>
      </header>

      {/* Live AI Analysis Control Panel */}
      <div className="bg-gradient-to-r from-blue-900 to-indigo-900 p-6 rounded-xl shadow-lg border border-indigo-200 mb-8 text-white">
        <h2 className="text-xl font-bold mb-3 flex items-center"><Zap className="mr-2 h-5 w-5 text-yellow-400"/> Live NLP Prediction Engine</h2>
        <p className="text-indigo-200 text-sm mb-4">Enter a Google Play App ID (e.g. <code>com.zing.zalo</code>, <code>vn.tiki.app.tikiandroid</code>, <code>com.shopee.vn</code>) to scrape and preprocess data via text normalization & tokenization in real-time.</p>
        
        <div className="flex flex-col md:flex-row gap-4">
          <input 
            type="text" 
            placeholder="App Package ID..." 
            className="flex-1 w-full bg-white/10 border border-indigo-400/30 text-white placeholder-indigo-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-yellow-400 focus:outline-none"
            value={appIdInput}
            onChange={e => setAppIdInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleLiveAnalysis()}
          />
          <button 
            onClick={handleLiveAnalysis}
            className="bg-yellow-500 hover:bg-yellow-400 text-indigo-900 font-bold px-6 py-2 rounded-lg transition-colors shadow-sm"
          >
            Run NLP Analysis
          </button>
          {isLiveMode && (
            <button 
              onClick={() => { setAppIdInput(""); fetchStaticData(); }}
              className="bg-indigo-600 hover:bg-indigo-500 text-white font-bold px-6 py-2 rounded-lg transition-colors shadow-sm border border-indigo-400"
            >
              View Global Dashboard
            </button>
          )}
        </div>
      </div>

      {loading ? (
         <div className="flex flex-col justify-center items-center h-64 text-slate-500">
           <Zap className="h-8 w-8 text-blue-500 animate-pulse mb-4" />
           <p className="text-xl font-semibold">Running Preprocessing, Regex Clean, and Term Frequency Analysis...</p>
         </div>
      ) : liveError ? (
         <div className="flex justify-center items-center h-64 text-red-500">
            <ServerCrash className="h-6 w-6 mr-2" /> {liveError}
         </div>
      ) : (
      <>
        {/* NEW NLP Analytics Panel */}
        {nlpInsights && (
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 mb-8">
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100 xl:col-span-1 border-t-4 border-t-red-500">
              <h2 className="text-lg font-bold mb-4 flex items-center text-slate-800"><Hash className="mr-2 h-5 w-5 text-red-500"/> Pain Points (Top Negatives)</h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={nlpInsights.top_negative_words} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <XAxis type="number" hide />
                    <YAxis dataKey="word" type="category" width={60} axisLine={false} tickLine={false} tick={{fontSize: 12}} />
                    <Tooltip cursor={{fill: '#fee2e2'}} />
                    <Bar dataKey="count" fill="#ef4444" radius={[0, 4, 4, 0]} barSize={20} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100 xl:col-span-1 border-t-4 border-t-green-500">
              <h2 className="text-lg font-bold mb-4 flex items-center text-slate-800"><Hash className="mr-2 h-5 w-5 text-green-500"/> Product Highlights (Positives)</h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={nlpInsights.top_positive_words} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <XAxis type="number" hide />
                    <YAxis dataKey="word" type="category" width={60} axisLine={false} tickLine={false} tick={{fontSize: 12}} />
                    <Tooltip cursor={{fill: '#dcfce3'}} />
                    <Bar dataKey="count" fill="#10b981" radius={[0, 4, 4, 0]} barSize={20} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100 xl:col-span-1 border-t-4 border-t-indigo-500">
              <h2 className="text-lg font-bold mb-4 flex items-center text-slate-800"><Activity className="mr-2 h-5 w-5 text-indigo-500"/> Rating Time-Series Trend</h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={nlpInsights.time_series_trend} margin={{ top: 20, right: 10, left: -20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                    <XAxis dataKey="date" tick={{fontSize: 12}} tickLine={false} />
                    <YAxis domain={[1, 5]} tick={{fontSize: 12}} tickLine={false} axisLine={false} />
                    <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                    <Line type="monotone" dataKey="average_score" stroke="#4f46e5" strokeWidth={3} dot={{r: 4}} activeDot={{r: 6}} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-8">
          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
            <h2 className="text-xl font-bold mb-4 flex items-center text-slate-800"><TrendingUp className="mr-2 h-5 w-5 text-blue-500"/> Overall Polarity Score</h2>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={barData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                  <XAxis type="number" domain={[-1, 1]} />
                  <YAxis dataKey="name" type="category" width={100} />
                  <Tooltip cursor={{fill: '#f1f5f9'}} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                  <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
            <h2 className="text-xl font-bold mb-4 flex items-center text-slate-800"><MessageSquare className="mr-2 h-5 w-5 text-green-500"/> Market Sentiment Breakdown</h2>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={pieData} cx="50%" cy="50%" innerRadius={70} outerRadius={110} paddingAngle={5} dataKey="value" label>
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={PIE_COLORS[entry.name] || '#ccc'} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                  <Legend verticalAlign="bottom" height={36}/>
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Reviews Table */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100 mb-8">
          <div className="flex flex-col md:flex-row gap-4 mb-6">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-3 h-5 w-5 text-slate-400" />
              <input 
                type="text" 
                placeholder="Search extracted reviews..." 
                className="w-full pl-10 pr-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 text-slate-700" 
                value={searchTerm} 
                onChange={e => setSearchTerm(e.target.value)} 
              />
            </div>
            
            <div className="flex gap-4">
              <select 
                className="px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 bg-white text-slate-700 font-medium"
                value={appFilter}
                onChange={e => setAppFilter(e.target.value)}
              >
                <option value="">All Apps</option>
                {uniqueCompanies.map(c => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>

              <select 
                className="px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 bg-white text-slate-700 font-medium"
                value={sentimentFilter}
                onChange={e => setSentimentFilter(e.target.value)}
              >
                <option value="">All Sentiments</option>
                <option value="Positive">Positive</option>
                <option value="Negative">Negative</option>
                <option value="Neutral">Neutral</option>
              </select>
            </div>
          </div>

          <h2 className="text-xl font-bold text-slate-800 mb-4 flex justify-between">
            <span>Raw Text Extraction & Preprocessing</span>
            <span className="text-sm font-medium text-blue-600 bg-blue-50 px-3 py-1 rounded-full">{filteredReviews.length} records processed</span>
          </h2>
          <div className="overflow-x-auto border border-slate-200 rounded-xl">
            <table className="w-full text-left">
              <thead className="bg-slate-100 text-slate-600">
                <tr>
                  <th className="p-4 font-semibold uppercase tracking-wider text-xs w-1/4">Target App</th>
                  <th className="p-4 font-semibold uppercase tracking-wider text-xs">Review Text & NLP Tokens</th>
                  <th className="p-4 font-semibold uppercase tracking-wider text-xs">AI Sentiment</th>
                </tr>
              </thead>
              <tbody>
                {filteredReviews.length === 0 ? (
                  <tr><td colSpan="3" className="text-center p-8 text-slate-500 italic">No reviews match your filters.</td></tr>
                ) : (
                  filteredReviews.slice(0, 100).map((r, i) => (
                    <tr key={i} className="border-t hover:bg-slate-50">
                      <td className="p-4 font-bold text-slate-900">{r.company}</td>
                      <td className="p-4">
                         <div className="italic text-slate-700 font-serif mb-2">"{r.review_text}"</div>
                         {r.cleaned_text && (
                           <div className="text-xs text-slate-500 font-mono bg-slate-100 p-2 rounded break-all">
                             <span className="font-semibold text-slate-400">Extracted Features:</span> [{r.cleaned_text}]
                           </div>
                         )}
                      </td>
                      <td className="p-4 text-center whitespace-nowrap">
                        <div className={`inline-flex flex-col items-center px-3 py-1 rounded w-32 ${r.predicted_sentiment === 'Positive' ? 'bg-green-100 text-green-700' : r.predicted_sentiment === 'Negative' ? 'bg-red-100 text-red-700' : 'bg-yellow-100 text-yellow-700'}`}>
                          <span className="font-bold text-sm">{r.predicted_sentiment}</span>
                          <span className="text-xs opacity-80 mt-1">Score: {parseFloat(r.polarity_score || r.true_score || 0).toFixed(2)}</span>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </>
      )}
    </div>
  );
}

export default App;
